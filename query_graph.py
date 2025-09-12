# ------------------------------
# FILE: query_graph.py
# ------------------------------
"""
Script to query per-event graphs and merged graphs to answer column-level lineage across jobs.
This patched version focuses on **upstream** BFS and includes job+namespace+resolved dataset in output.
"""

from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Iterable
import networkx as nx
from networkx.readwrite import gpickle
import sys

_LOG = logging.getLogger("query_graph")


def setup_logging(level=logging.INFO):
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    _LOG.addHandler(h)
    _LOG.setLevel(level)


def load_graph(path: Path) -> nx.DiGraph:
    if not path.exists():
        raise FileNotFoundError(f"Graph file not found: {path}")
    return gpickle.read_gpickle(str(path))


def find_dataset_column_nodes(graph: nx.DiGraph, dataset: str, column: str, case_insensitive: bool = True) -> List[str]:
    dataset_q = dataset.lower() if case_insensitive else dataset
    column_q = column.lower() if case_insensitive else column
    matches = []
    for n, attrs in graph.nodes(data=True):
        if not isinstance(n, str):
            continue
        parsed = attrs.get("parsed") or {}
        ds = (parsed.get("dataset") or "")
        col = (parsed.get("column") or "")
        if case_insensitive:
            ds = ds.lower(); col = col.lower()
        if dataset_q == ds and column_q == col:
            matches.append(n)
    return matches


def bfs_upstream(graph: nx.DiGraph, start_nodes: Iterable[str], max_hops: Optional[int] = None) -> Dict[str, Any]:
    """
    Traverse upstream lineage (origins) for given start nodes.
    Returns node metadata including job names, namespaces, and resolved dataset info.
    """
    from collections import deque
    visited = set()
    q = deque()
    results = []

    for s in start_nodes:
        q.append((s, 0, None))  # node, depth, parent

    while q:
        node, depth, parent = q.popleft()
        if node in visited:
            continue
        visited.add(node)

        node_data = graph.nodes.get(node, {})
        parsed = node_data.get("parsed", {})

        results.append({
            "node": node,
            "depth": depth,
            "parsed": parsed,
            "runs": node_data.get("runs", []),
            "jobs": node_data.get("jobs", []),
            "namespaces": node_data.get("namespaces", []),
            "resolved_dataset": parsed.get("resolved_dataset"),
            "resolved_column": parsed.get("resolved_column"),
            "parent": parent
        })

        if max_hops is not None and depth >= max_hops:
            continue

        for pred in graph.predecessors(node):
            q.append((pred, depth + 1, node))

    return {"visited_nodes": list(visited), "paths": results}


def query_from_merged_graph(merged_graph: nx.DiGraph, dataset: str, column: str, max_hops: Optional[int] = None, case_insensitive: bool = True) -> Dict[str, Any]:
    start_nodes = find_dataset_column_nodes(merged_graph, dataset, column, case_insensitive=case_insensitive)
    if not start_nodes:
        return {"error": f"No matching nodes found for dataset={dataset} column={column}"}
    trace = bfs_upstream(merged_graph, start_nodes, max_hops=max_hops)
    return {"start_nodes": start_nodes, "trace": trace}


def merge_graphs(graph_paths: Iterable[Path]) -> nx.DiGraph:
    graphs = [load_graph(p) for p in graph_paths]
    merged = nx.DiGraph()
    for g in graphs:
        merged = nx.compose(merged, g)
    return merged


def query_using_index(index_path: Path, per_event_dir: Path, dataset: str, column: str, merge_selected: bool = False, max_hops: Optional[int] = None) -> Dict[str, Any]:
    if not index_path.exists():
        raise FileNotFoundError(f"Index file not found: {index_path}")
    with index_path.open("r", encoding="utf-8") as f:
        index = json.load(f)
    entries = index.get(dataset) or []
    if not entries:
        return {"error": f"No events found referencing dataset '{dataset}' in index"}
    graph_files = []
    for e in entries:
        gf = e.get("graph_file")
        if not gf:
            continue
        p = Path(gf)
        if not p.is_absolute():
            p = per_event_dir / p
        if p.exists():
            graph_files.append(p)
    if not graph_files:
        return {"error": f"No per-event graph files found for dataset '{dataset}'"}

    if merge_selected:
        merged = merge_graphs(graph_files)
        return query_from_merged_graph(merged, dataset, column, max_hops=max_hops)

    all_results = {"per_event": {}}
    for gf in graph_files:
        g = load_graph(gf)
        res = query_from_merged_graph(g, dataset, column, max_hops=max_hops)
        all_results["per_event"][str(gf)] = res
    return all_results


def main(argv=None):
    parser = argparse.ArgumentParser(prog="query_graph", description="Query column lineage using per-event or merged graphs (upstream only)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--graph", type=Path, help="Path to merged graph gpickle")
    group.add_argument("--index", type=Path, help="Path to dataset_event_index.json (per-event graphs)" )

    parser.add_argument("--per-event-dir", type=Path, default=Path("./per_event_graphs"), help="Directory where per-event graphs live (used with --index)")
    parser.add_argument("--dataset", type=str, required=True, help="Target dataset name (as parsed from nodes)")
    parser.add_argument("--column", type=str, required=True, help="Target column name")
    parser.add_argument("--max-hops", type=int, default=None, help="Maximum upstream hops to traverse")
    parser.add_argument("--merge-selected", action="store_true", help="When using --index, merge selected per-event graphs into a single graph before querying")
    parser.add_argument("--case-insensitive", action="store_true", default=True, help="Match dataset/column case-insensitively")
    parser.add_argument("--out", type=Path, default=None, help="Optional output file to write JSON results")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args(argv)
    setup_logging(logging.DEBUG if args.debug else logging.INFO)

    if args.graph:
        _LOG.info("Loading merged graph from %s", args.graph)
        g = load_graph(args.graph)
        res = query_from_merged_graph(g, args.dataset, args.column, max_hops=args.max_hops, case_insensitive=args.case_insensitive)
    else:
        _LOG.info("Using index %s to locate per-event graphs", args.index)
        res = query_using_index(args.index, args.per_event_dir, args.dataset, args.column, merge_selected=args.merge_selected, max_hops=args.max_hops)

    out_json = json.dumps(res, indent=2)
    if args.out:
        out_path = Path(args.out)
        if out_path.parent:
            out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(out_json, encoding="utf-8")
        _LOG.info("Wrote results to %s", out_path)
    else:
        print(out_json)


if __name__ == "__main__":
    main()

# FILE: build_event_graphs.py
"""
Production-grade script to build per-event NetworkX graphs from an OpenLineage-like
response.json file and optionally build a merged/global graph.

This patched version handles:
 - networkx >=3 gpickle API
 - `table_paths` that may be a graph-shaped dict with `nodes`/`edges`
 - operationField resolution from complex `table_paths` node shapes
 - job name/namespace capture with fallbacks
 - safer output file handling

Outputs:
 - per-event gpickle files in output directory (default: ./per_event_graphs)
 - index.json mapping dataset -> list of event indexes/run_ids that reference it
 - optional merged graph (gpickle)

Usage (CLI):
    python build_event_graphs.py --response /mnt/data/response.json --out-dir ./per_event_graphs --force

"""

from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Iterable, Optional
import networkx as nx
from networkx.readwrite import gpickle
import sys

_LOG = logging.getLogger("build_event_graphs")


def setup_logging(level=logging.INFO):
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    _LOG.addHandler(h)
    _LOG.setLevel(level)


def load_response_json(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"response.json not found at: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # Expect top-level to contain an 'events' list (openlineage style) or be a list itself
    if isinstance(data, dict) and "events" in data and isinstance(data["events"], list):
        return data["events"]
    if isinstance(data, list):
        return data
    raise ValueError("response.json must contain a list of events or an object with 'events' list")


def normalize_text(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    return s.strip()


def parse_field_node(node_str: str) -> Dict[str, Optional[str]]:
    out = {"raw": node_str, "kind": None, "namespace": None, "dataset": None, "column": None}
    if not node_str or not isinstance(node_str, str):
        return out
    parts = node_str.split(":::")
    if len(parts) >= 4 and parts[0].lower().startswith("datasetfield"):
        out["kind"] = "datasetField"
        out["namespace"] = normalize_text(parts[1])
        out["dataset"] = normalize_text(parts[2])
        out["column"] = normalize_text(":::".join(parts[3:]))
    elif len(parts) >= 2 and parts[0].lower().startswith("operationfield"):
        # operationField:::namespace:::operationId:::column
        out["kind"] = "operationField"
        if len(parts) >= 4:
            out["namespace"] = normalize_text(parts[1])
            out["dataset"] = normalize_text(parts[2])  # holds operation id
            out["column"] = normalize_text(":::".join(parts[3:]))
    elif len(parts) >= 2:
        out["kind"] = parts[0]
        if len(parts) >= 3:
            out["namespace"] = normalize_text(parts[1])
            out["dataset"] = normalize_text(parts[2])
            if len(parts) >= 4:
                out["column"] = normalize_text(parts[3])
    return out


def event_graph_id(ev: Dict[str, Any], index: Optional[int] = None) -> str:
    runid = ev.get("run", {}).get("runId")
    if runid:
        return str(runid)
    if index is not None:
        return f"event_{index}"
    et = ev.get("eventTime", "no_time")
    prod = ev.get("producer", "no_producer")
    return f"{prod}_{et}"


def _iter_column_paths_from_event(ev: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    run_facets = ev.get("run", {}).get("facets", {})
    trans = run_facets.get("transformations") or run_facets.get("transformation") or {}
    col_paths = []
    if isinstance(trans, dict):
        col_paths = trans.get("column_paths") or trans.get("columnPaths") or []
    if not isinstance(col_paths, list):
        col_paths = []
    return col_paths


def ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def build_graph_from_event(ev: Dict[str, Any], index: Optional[int] = None, normalize_nodes: bool = True) -> nx.DiGraph:
    """Build a directed graph from a single event's column_paths.

    Robust to multiple `table_paths` shapes (list or graph 'nodes/edges').
    """
    G = nx.DiGraph()
    runid = ev.get("run", {}).get("runId")

    # --- Capture job info with fallbacks ---
    job_info = ev.get("job") or {}
    job_name = job_info.get("name") or ev.get("run", {}).get("facets", {}).get("job", {}).get("name") or ev.get("producer")
    job_ns = job_info.get("namespace") or ev.get("run", {}).get("facets", {}).get("job", {}).get("namespace") or ev.get("run", {}).get("producer")

    trans = ev.get("run", {}).get("facets", {}).get("transformations", {}) or {}
    col_paths = trans.get("column_paths") or []
    table_paths = trans.get("table_paths") or trans.get("tablePaths") or []

    # --- Build operation → dataset map (robust) ---
    op_map: Dict[str, Dict[str, Optional[str]]] = {}
    if isinstance(table_paths, dict):
        nodes = table_paths.get("nodes") or []
        for node in nodes:
            if not isinstance(node, dict):
                continue
            data = node.get("data") or {}
            data_id = data.get("id") or {}
            mapped_dataset = node.get("label") or data.get("physicalName") or None
            mapped_namespace = node.get("namespace") or (data_id.get("namespace") if isinstance(data_id, dict) else None)

            op_key_candidates = []
            # try nested id name
            if isinstance(data_id, dict):
                name = data_id.get("name")
                ns = data_id.get("namespace")
                if name:
                    op_key_candidates.append(name)
                    if ns:
                        op_key_candidates.append(f"{ns}::{name}")
                        op_key_candidates.append(f"operation::{ns}::{name}")
            # fallback to top-level
            name2 = node.get("name") or node.get("id")
            if isinstance(name2, str):
                op_key_candidates.append(name2)
            # register
            for k in op_key_candidates:
                if k and k not in op_map:
                    op_map[k] = {"dataset": mapped_dataset, "namespace": mapped_namespace, "raw_node": node}
    elif isinstance(table_paths, list):
        for tp in table_paths:
            if not isinstance(tp, dict):
                continue
            op = tp.get("operation")
            if not op:
                continue
            op_map[op] = {"dataset": tp.get("dataset"), "namespace": tp.get("namespace"), "raw": tp}

    # helper to stringify nodes
    def stringify(x):
        if isinstance(x, str):
            return x
        if isinstance(x, dict):
            for k in ("field", "column", "name", "datasetField"):
                if k in x and isinstance(x[k], str):
                    return x[k]
            try:
                return json.dumps(x, ensure_ascii=False)
            except Exception:
                return str(x)
        return str(x)

    def parse_and_resolve(node_str: str):
        parsed = parse_field_node(node_str)
        if parsed.get("kind") == "operationField":
            op_candidate = parsed.get("dataset")
            ns_candidate = parsed.get("namespace")
            candidates = []
            if op_candidate:
                candidates.append(op_candidate)
                if ns_candidate:
                    candidates.append(f"{ns_candidate}::{op_candidate}")
                    candidates.append(f"operation::{ns_candidate}::{op_candidate}")
                candidates.append(node_str)
            resolved = None
            for c in candidates:
                if c and c in op_map:
                    resolved = op_map[c]
                    break
            if resolved:
                parsed["resolved_dataset"] = resolved.get("dataset")
                parsed["resolved_namespace"] = resolved.get("namespace")
                parsed["resolved_column"] = parsed.get("column")
        return parsed

    # Build edges
    for cp in col_paths:
        origins = ensure_list(cp.get("origin") or cp.get("origins"))
        dests = ensure_list(cp.get("destination") or cp.get("destinations") or cp.get("destinationColumns"))

        origins = [stringify(o) for o in origins]
        dests = [stringify(d) for d in dests]

        for o in origins:
            if not o:
                continue
            key_o = o.strip() if normalize_nodes else o
            parsed_o = parse_and_resolve(key_o)

            if not G.has_node(key_o):
                G.add_node(key_o,
                           parsed=parsed_o,
                           runs=[runid] if runid else [],
                           jobs=[job_name] if job_name else [],
                           namespaces=[job_ns] if job_ns else [])
            else:
                if runid and runid not in G.nodes[key_o].get("runs", []):
                    G.nodes[key_o].setdefault("runs", []).append(runid)
                if job_name and job_name not in G.nodes[key_o].get("jobs", []):
                    G.nodes[key_o].setdefault("jobs", []).append(job_name)
                if job_ns and job_ns not in G.nodes[key_o].get("namespaces", []):
                    G.nodes[key_o].setdefault("namespaces", []).append(job_ns)

            for d in dests:
                if not d:
                    continue
                key_d = d.strip() if normalize_nodes else d
                parsed_d = parse_and_resolve(key_d)

                if not G.has_node(key_d):
                    G.add_node(key_d,
                               parsed=parsed_d,
                               runs=[runid] if runid else [],
                               jobs=[job_name] if job_name else [],
                               namespaces=[job_ns] if job_ns else [])
                else:
                    if runid and runid not in G.nodes[key_d].get("runs", []):
                        G.nodes[key_d].setdefault("runs", []).append(runid)
                    if job_name and job_name not in G.nodes[key_d].get("jobs", []):
                        G.nodes[key_d].setdefault("jobs", []).append(job_name)
                    if job_ns and job_ns not in G.nodes[key_d].get("namespaces", []):
                        G.nodes[key_d].setdefault("namespaces", []).append(job_ns)

                edge_meta = {
                    "run_id": runid,
                    "job_name": job_name,
                    "job_namespace": job_ns,
                    "event_index": index,
                    "cp_summary": {k: cp.get(k) for k in ("transformation", "expression") if k in cp}
                }

                if G.has_edge(key_o, key_d):
                    G.edges[key_o, key_d].setdefault("events", []).append(edge_meta)
                else:
                    G.add_edge(key_o, key_d, events=[edge_meta])

    return G


def save_graph(G: nx.DiGraph, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    gpickle.write_gpickle(G, str(path))


def load_graph(path: Path) -> nx.DiGraph:
    return gpickle.read_gpickle(str(path))


def compose_graphs(graphs: Iterable[nx.DiGraph]) -> nx.DiGraph:
    merged = nx.DiGraph()
    for g in graphs:
        merged = nx.compose(merged, g)
    for u, v, data in merged.edges(data=True):
        if "events" not in data:
            data["events"] = data.get("events", [])
    return merged


def build_all_event_graphs(response_path: Path, out_dir: Path, force: bool = False, build_merged: bool = False, merged_path: Optional[Path] = None) -> Dict[str, Any]:
    events = load_response_json(response_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_map: Dict[str, List[Dict[str, Any]]] = {}
    saved_graph_paths: List[Path] = []

    _LOG.info("Found %d events in response file", len(events))

    for i, ev in enumerate(events):
        try:
            gid = event_graph_id(ev, index=i)
            safe_gid = gid.replace("/", "_").replace("\\", "_")
            fname = out_dir / f"event_{i}__{safe_gid}.gpickle"
            if fname.exists() and not force:
                _LOG.debug("Skipping existing graph %s (use --force to rebuild)", fname)
                saved_graph_paths.append(fname)
                g = load_graph(fname)
            else:
                _LOG.info("Building graph for event index %d (id=%s)", i, gid)
                g = build_graph_from_event(ev, index=i, normalize_nodes=True)
                save_graph(g, fname)
                saved_graph_paths.append(fname)

            # update dataset->events index map by scanning nodes for datasetField pattern
            for n, attrs in g.nodes(data=True):
                parsed = attrs.get("parsed") or {}
                if parsed.get("kind") == "datasetField" and parsed.get("dataset"):
                    ds = parsed.get("dataset")
                    entry = {"event_index": i, "run_id": ev.get("run", {}).get("runId"), "graph_file": str(fname)}
                    index_map.setdefault(ds, []).append(entry)
        except Exception as e:
            _LOG.exception("Failed to build graph for event index %s: %s", i, e)
            continue

    index_path = out_dir / "dataset_event_index.json"
    with index_path.open("w", encoding="utf-8") as f:
        json.dump(index_map, f, indent=2)
    _LOG.info("Wrote dataset_event_index.json with %d datasets", len(index_map))

    if build_merged:
        _LOG.info("Building merged graph from %d per-event graphs", len(saved_graph_paths))
        graphs = [load_graph(p) for p in saved_graph_paths]
        merged = compose_graphs(graphs)
        if merged_path is None:
            merged_path = out_dir / "merged_graph.gpickle"
        save_graph(merged, merged_path)
        _LOG.info("Wrote merged graph to %s (nodes=%d edges=%d)", merged_path, merged.number_of_nodes(), merged.number_of_edges())

    return {"saved_graphs": [str(p) for p in saved_graph_paths], "index_file": str(index_path)}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="build_event_graphs", description="Build per-event lineage graphs from response.json")
    parser.add_argument("--response", type=Path, default=Path("/mnt/data/response.json"), help="Path to response.json")
    parser.add_argument("--out-dir", type=Path, default=Path("./per_event_graphs"), help="Output directory for graphs and index")
    parser.add_argument("--force", action="store_true", help="Force rebuild of graphs even if files exist")
    parser.add_argument("--build-merged", action="store_true", help="Also build a merged global graph")
    parser.add_argument("--merged-path", type=Path, default=None, help="Path for merged graph (if --build-merged)")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args(argv)
    setup_logging(logging.DEBUG if args.debug else logging.INFO)

    try:
        result = build_all_event_graphs(args.response, args.out_dir, force=args.force, build_merged=args.build_merged, merged_path=args.merged_path)
        _LOG.info("Done. Summary: %s", result)
    except Exception:
        _LOG.exception("Fatal error while building event graphs")
        raise


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
OpenLineage FastMCP Server for comprehensive data lineage analysis.

This FastMCP server provides tools to query column lineage, table lineage, 
and trace data dependencies across events, jobs, and unlimited depths.

Updated to match the enhanced build_graph.py and query_graph.py functionality.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple
import networkx as nx
from networkx.readwrite import gpickle
from collections import defaultdict, deque

# FastMCP imports
from fastmcp import FastMCP

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("openlineage_mcp")

def normalize_text(s: Optional[str]) -> Optional[str]:
    """Normalize text by stripping whitespace."""
    if s is None:
        return None
    return s.strip()

def parse_field_node(node_str: str) -> Dict[str, Optional[str]]:
    """Parse field node string into components (matches build_graph.py logic)."""
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

class OpenLineageAnalyzer:
    """Core analyzer class with enhanced querying capabilities."""
    
    def __init__(self):
        self.merged_graph: Optional[nx.DiGraph] = None
        self.per_event_graphs: Dict[str, nx.DiGraph] = {}
        self.dataset_index: Dict[str, List[Dict[str, Any]]] = {}
        self.table_index: Dict[str, Set[str]] = defaultdict(set)  # table -> columns
        self.column_index: Dict[str, Set[str]] = defaultdict(set)  # column -> tables
        
    def load_merged_graph(self, path: Path) -> None:
        """Load a merged graph from gpickle file using updated API."""
        if not path.exists():
            raise FileNotFoundError(f"Graph file not found: {path}")
        self.merged_graph = gpickle.read_gpickle(str(path))
        self._build_indices()
        logger.info(f"Loaded merged graph with {self.merged_graph.number_of_nodes()} nodes, {self.merged_graph.number_of_edges()} edges")
    
    def load_per_event_graphs(self, index_path: Path, graphs_dir: Path) -> None:
        """Load per-event graphs using the dataset index."""
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
            
        with index_path.open("r", encoding="utf-8") as f:
            self.dataset_index = json.load(f)
        
        # Load all referenced graph files
        graph_files = set()
        for entries in self.dataset_index.values():
            for entry in entries:
                graph_file = entry.get("graph_file")
                if graph_file:
                    path = Path(graph_file)
                    if not path.is_absolute():
                        path = graphs_dir / path
                    if path.exists():
                        graph_files.add(path)
        
        for path in graph_files:
            try:
                graph = gpickle.read_gpickle(str(path))
                self.per_event_graphs[str(path)] = graph
                logger.info(f"Loaded graph {path} with {graph.number_of_nodes()} nodes")
            except Exception as e:
                logger.error(f"Failed to load graph {path}: {e}")
        
        self._build_indices()
    
    def _build_indices(self) -> None:
        """Build internal indices for fast lookups."""
        graphs = []
        if self.merged_graph:
            graphs.append(self.merged_graph)
        else:
            graphs.extend(self.per_event_graphs.values())
        
        for graph in graphs:
            for node, attrs in graph.nodes(data=True):
                parsed = attrs.get("parsed", {})
                if parsed.get("kind") == "datasetField":
                    dataset = parsed.get("dataset")
                    column = parsed.get("column")
                    if dataset and column:
                        self.table_index[dataset].add(column)
                        self.column_index[column].add(dataset)
                # Also handle resolved datasets from operationField
                elif parsed.get("kind") == "operationField":
                    resolved_dataset = parsed.get("resolved_dataset")
                    resolved_column = parsed.get("resolved_column")
                    if resolved_dataset and resolved_column:
                        self.table_index[resolved_dataset].add(resolved_column)
                        self.column_index[resolved_column].add(resolved_dataset)
    
    def find_nodes_by_pattern(self, dataset: Optional[str] = None, column: Optional[str] = None, 
                            case_insensitive: bool = True) -> List[str]:
        """Find nodes matching dataset/column patterns with enhanced parsing."""
        graph = self.merged_graph or nx.compose_all(self.per_event_graphs.values())
        matches = []
        
        for node, attrs in graph.nodes(data=True):
            parsed = attrs.get("parsed", {})
            
            # Check both direct dataset/column and resolved versions
            node_datasets = []
            node_columns = []
            
            if parsed.get("dataset"):
                node_datasets.append(parsed["dataset"])
            if parsed.get("resolved_dataset"):
                node_datasets.append(parsed["resolved_dataset"])
                
            if parsed.get("column"):
                node_columns.append(parsed["column"])
            if parsed.get("resolved_column"):
                node_columns.append(parsed["resolved_column"])
            
            if case_insensitive:
                node_datasets = [d.lower() for d in node_datasets]
                node_columns = [c.lower() for c in node_columns]
                match_dataset = dataset.lower() if dataset else None
                match_column = column.lower() if column else None
            else:
                match_dataset = dataset
                match_column = column
            
            dataset_match = not match_dataset or any(nd == match_dataset for nd in node_datasets)
            column_match = not match_column or any(nc == match_column for nc in node_columns)
            
            if dataset_match and column_match:
                matches.append(node)
        
        return matches
    
    def bfs_upstream(self, start_nodes: List[str], max_hops: Optional[int] = None) -> Dict[str, Any]:
        """
        Traverse upstream lineage (origins) for given start nodes.
        Returns node metadata including job names, namespaces, and resolved dataset info.
        Matches the logic from query_graph.py.
        """
        graph = self.merged_graph or nx.compose_all(self.per_event_graphs.values())
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
    
    def trace_lineage(self, start_nodes: List[str], direction: str = "upstream", 
                     max_depth: Optional[int] = None, include_metadata: bool = True) -> Dict[str, Any]:
        """Trace lineage upstream or downstream from starting nodes."""
        if direction == "upstream":
            return self.bfs_upstream(start_nodes, max_depth)
            
        graph = self.merged_graph or nx.compose_all(self.per_event_graphs.values())
        
        if direction not in ["upstream", "downstream"]:
            raise ValueError("Direction must be 'upstream' or 'downstream'")
        
        visited = set()
        paths = []
        queue = deque([(node, 0, [node]) for node in start_nodes])
        
        while queue:
            current_node, depth, path = queue.popleft()
            
            if current_node in visited:
                continue
            visited.add(current_node)
            
            node_data = graph.nodes.get(current_node, {})
            parsed = node_data.get("parsed", {})
            
            path_entry = {
                "node": current_node,
                "depth": depth,
                "path": path.copy(),
                "parsed": parsed,
                "runs": node_data.get("runs", []),
                "jobs": node_data.get("jobs", []),
                "namespaces": node_data.get("namespaces", []),
                "resolved_dataset": parsed.get("resolved_dataset"),
                "resolved_column": parsed.get("resolved_column")
            }
            
            if include_metadata:
                path_entry["metadata"] = {
                    "in_degree": graph.in_degree(current_node),
                    "out_degree": graph.out_degree(current_node)
                }
            
            paths.append(path_entry)
            
            if max_depth is not None and depth >= max_depth:
                continue
            
            # Get neighbors based on direction
            if direction == "downstream":
                neighbors = list(graph.successors(current_node))
            else:
                neighbors = list(graph.predecessors(current_node))
            
            for neighbor in neighbors:
                new_path = path + [neighbor]
                queue.append((neighbor, depth + 1, new_path))
        
        return {
            "start_nodes": start_nodes,
            "direction": direction,
            "visited_count": len(visited),
            "paths": paths,
            "max_depth_reached": max([p["depth"] for p in paths]) if paths else 0
        }
    
    def find_data_flow_paths(self, source_dataset: str, target_dataset: str, 
                           max_paths: int = 10) -> Dict[str, Any]:
        """Find all paths between source and target datasets."""
        graph = self.merged_graph or nx.compose_all(self.per_event_graphs.values())
        
        source_nodes = self.find_nodes_by_pattern(dataset=source_dataset)
        target_nodes = self.find_nodes_by_pattern(dataset=target_dataset)
        
        if not source_nodes:
            return {"error": f"No nodes found for source dataset: {source_dataset}"}
        if not target_nodes:
            return {"error": f"No nodes found for target dataset: {target_dataset}"}
        
        all_paths = []
        for source in source_nodes:
            for target in target_nodes:
                try:
                    paths = list(nx.all_simple_paths(graph, source, target, cutoff=max_paths))
                    for path in paths[:max_paths]:
                        path_info = {
                            "source": source,
                            "target": target,
                            "path": path,
                            "length": len(path) - 1,
                            "nodes_detail": []
                        }
                        
                        for node in path:
                            node_data = graph.nodes.get(node, {})
                            parsed = node_data.get("parsed", {})
                            path_info["nodes_detail"].append({
                                "node": node,
                                "parsed": parsed,
                                "runs": node_data.get("runs", []),
                                "jobs": node_data.get("jobs", []),
                                "namespaces": node_data.get("namespaces", []),
                                "resolved_dataset": parsed.get("resolved_dataset"),
                                "resolved_column": parsed.get("resolved_column")
                            })
                        
                        all_paths.append(path_info)
                except nx.NetworkXNoPath:
                    continue
        
        return {
            "source_dataset": source_dataset,
            "target_dataset": target_dataset,
            "paths_found": len(all_paths),
            "paths": all_paths[:max_paths]
        }
    
    def analyze_column_impact(self, dataset: str, column: str) -> Dict[str, Any]:
        """Analyze the impact of changes to a specific column."""
        downstream = self.trace_lineage(
            self.find_nodes_by_pattern(dataset=dataset, column=column),
            direction="downstream"
        )
        
        affected_tables = set()
        affected_columns = set()
        affected_jobs = set()
        
        for path in downstream["paths"]:
            parsed = path["parsed"]
            # Check both direct and resolved datasets/columns
            datasets = [parsed.get("dataset"), parsed.get("resolved_dataset")]
            columns = [parsed.get("column"), parsed.get("resolved_column")]
            
            for ds in datasets:
                if ds:
                    affected_tables.add(ds)
            for col in columns:
                if col:
                    affected_columns.add(col)
            
            for job in path.get("jobs", []):
                if job:
                    affected_jobs.add(job)
            for run in path.get("runs", []):
                if run:
                    affected_jobs.add(run)
        
        return {
            "source_column": f"{dataset}.{column}",
            "downstream_analysis": downstream,
            "impact_summary": {
                "affected_tables": list(affected_tables),
                "affected_columns": list(affected_columns),
                "affected_jobs": list(affected_jobs),
                "total_downstream_nodes": downstream["visited_count"]
            }
        }
    
    def get_table_schema_evolution(self, dataset: str) -> Dict[str, Any]:
        """Track schema evolution for a table across events."""
        if not self.dataset_index:
            return {"error": "Per-event data not loaded"}
        
        entries = self.dataset_index.get(dataset, [])
        schema_evolution = []
        
        for entry in entries:
            graph_file = entry["graph_file"]
            if graph_file in self.per_event_graphs:
                graph = self.per_event_graphs[graph_file]
                columns = set()
                
                for node, attrs in graph.nodes(data=True):
                    parsed = attrs.get("parsed", {})
                    # Check both direct and resolved datasets
                    node_datasets = [parsed.get("dataset"), parsed.get("resolved_dataset")]
                    node_columns = [parsed.get("column"), parsed.get("resolved_column")]
                    
                    if dataset in node_datasets:
                        for col in node_columns:
                            if col:
                                columns.add(col)
                
                schema_evolution.append({
                    "event_index": entry.get("event_index"),
                    "run_id": entry.get("run_id"),
                    "columns": sorted(list(columns)),
                    "column_count": len(columns)
                })
        
        return {
            "dataset": dataset,
            "schema_evolution": sorted(schema_evolution, key=lambda x: x.get("event_index", 0)),
            "total_events": len(schema_evolution)
        }

# Initialize the FastMCP server and analyzer
mcp = FastMCP("OpenLineage Lineage Analysis")
analyzer = OpenLineageAnalyzer()

@mcp.tool()
def load_merged_graph(graph_path: str) -> str:
    """Load a merged OpenLineage graph from gpickle file.
    
    Args:
        graph_path: Path to the merged graph gpickle file
        
    Returns:
        Success message with graph statistics
    """
    try:
        path = Path(graph_path)
        analyzer.load_merged_graph(path)
        return (f"Successfully loaded merged graph from {graph_path}. "
               f"Nodes: {analyzer.merged_graph.number_of_nodes()}, "
               f"Edges: {analyzer.merged_graph.number_of_edges()}")
    except Exception as e:
        return f"Error loading merged graph: {str(e)}"

@mcp.tool()
def load_per_event_graphs(index_path: str, graphs_dir: str) -> str:
    """Load per-event OpenLineage graphs using dataset index.
    
    Args:
        index_path: Path to dataset_event_index.json file
        graphs_dir: Directory containing per-event graph files
        
    Returns:
        Success message with loading statistics
    """
    try:
        idx_path = Path(index_path)
        gr_dir = Path(graphs_dir)
        analyzer.load_per_event_graphs(idx_path, gr_dir)
        return (f"Successfully loaded {len(analyzer.per_event_graphs)} per-event graphs. "
               f"Tracking {len(analyzer.dataset_index)} datasets.")
    except Exception as e:
        return f"Error loading per-event graphs: {str(e)}"

@mcp.tool()
def trace_column_lineage(dataset: str, column: str, direction: str = "upstream", 
                        max_depth: Optional[int] = None, case_insensitive: bool = True) -> str:
    """Trace column-level lineage upstream or downstream.
    
    Args:
        dataset: Dataset/table name
        column: Column name
        direction: Trace direction ('upstream' or 'downstream')
        max_depth: Maximum depth to traverse (unlimited if not specified)
        case_insensitive: Case-insensitive matching
        
    Returns:
        JSON string with lineage trace results
    """
    try:
        if direction not in ["upstream", "downstream"]:
            return "Error: direction must be 'upstream' or 'downstream'"
            
        start_nodes = analyzer.find_nodes_by_pattern(dataset, column, case_insensitive)
        if not start_nodes:
            return f"No nodes found for {dataset}.{column}"
        
        result = analyzer.trace_lineage(start_nodes, direction, max_depth)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error tracing column lineage: {str(e)}"

@mcp.tool()
def trace_table_lineage(dataset: str, direction: str = "upstream", 
                       max_depth: Optional[int] = None, case_insensitive: bool = True) -> str:
    """Trace table-level lineage upstream or downstream.
    
    Args:
        dataset: Dataset/table name
        direction: Trace direction ('upstream' or 'downstream')
        max_depth: Maximum depth to traverse
        case_insensitive: Case-insensitive matching
        
    Returns:
        JSON string with table lineage results
    """
    try:
        if direction not in ["upstream", "downstream"]:
            return "Error: direction must be 'upstream' or 'downstream'"
            
        start_nodes = analyzer.find_nodes_by_pattern(dataset=dataset, case_insensitive=case_insensitive)
        if not start_nodes:
            return f"No nodes found for dataset {dataset}"
            
        result = analyzer.trace_lineage(start_nodes, direction, max_depth)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error tracing table lineage: {str(e)}"

@mcp.tool()
def find_data_flow_paths(source_dataset: str, target_dataset: str, max_paths: int = 10) -> str:
    """Find all data flow paths between source and target datasets.
    
    Args:
        source_dataset: Source dataset name
        target_dataset: Target dataset name
        max_paths: Maximum number of paths to return
        
    Returns:
        JSON string with all found paths
    """
    try:
        result = analyzer.find_data_flow_paths(source_dataset, target_dataset, max_paths)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error finding data flow paths: {str(e)}"

@mcp.tool()
def analyze_column_impact(dataset: str, column: str) -> str:
    """Analyze the downstream impact of changes to a specific column.
    
    Args:
        dataset: Dataset/table name
        column: Column name
        
    Returns:
        JSON string with impact analysis results
    """
    try:
        result = analyzer.analyze_column_impact(dataset, column)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error analyzing column impact: {str(e)}"

@mcp.tool()
def get_schema_evolution(dataset: str) -> str:
    """Track schema evolution for a table across events.
    
    Args:
        dataset: Dataset/table name
        
    Returns:
        JSON string with schema evolution history
    """
    try:
        result = analyzer.get_table_schema_evolution(dataset)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error getting schema evolution: {str(e)}"

@mcp.tool()
def search_columns(column_pattern: str, case_insensitive: bool = True) -> str:
    """Search for columns across all datasets.
    
    Args:
        column_pattern: Column name or pattern to search for
        case_insensitive: Case-insensitive search
        
    Returns:
        JSON string with search results grouped by dataset
    """
    try:
        matches = analyzer.find_nodes_by_pattern(column=column_pattern, case_insensitive=case_insensitive)
        
        # Group by dataset
        by_dataset = defaultdict(list)
        graph = analyzer.merged_graph or nx.compose_all(analyzer.per_event_graphs.values())
        
        for node in matches:
            parsed = graph.nodes.get(node, {}).get("parsed", {})
            # Check both direct and resolved datasets/columns
            datasets = [parsed.get("dataset"), parsed.get("resolved_dataset")]
            columns = [parsed.get("column"), parsed.get("resolved_column")]
            
            for dataset in datasets:
                if dataset:
                    for column in columns:
                        if column and column not in by_dataset[dataset]:
                            by_dataset[dataset].append(column)
        
        result = {
            "search_pattern": column_pattern,
            "total_matches": len(matches),
            "matches_by_dataset": dict(by_dataset)
        }
        
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error searching columns: {str(e)}"

@mcp.tool()
def get_lineage_summary() -> str:
    """Get a comprehensive lineage summary for analysis.
    
    Returns:
        JSON string with comprehensive lineage statistics
    """
    try:
        graph = analyzer.merged_graph or nx.compose_all(analyzer.per_event_graphs.values())
        
        # Basic graph stats
        total_nodes = graph.number_of_nodes()
        total_edges = graph.number_of_edges()
        
        # Count by type
        dataset_fields = 0
        operation_fields = 0
        datasets = set()
        columns = set()
        jobs = set()
        namespaces = set()
        
        for node, attrs in graph.nodes(data=True):
            parsed = attrs.get("parsed", {})
            node_kind = parsed.get("kind")
            
            if node_kind == "datasetField":
                dataset_fields += 1
                if parsed.get("dataset"):
                    datasets.add(parsed["dataset"])
                if parsed.get("column"):
                    columns.add(parsed["column"])
            elif node_kind == "operationField":
                operation_fields += 1
                if parsed.get("resolved_dataset"):
                    datasets.add(parsed["resolved_dataset"])
                if parsed.get("resolved_column"):
                    columns.add(parsed["resolved_column"])
            
            # Collect job and namespace info
            for job in attrs.get("jobs", []):
                if job:
                    jobs.add(job)
            for ns in attrs.get("namespaces", []):
                if ns:
                    namespaces.add(ns)
        
        summary = {
            "graph_statistics": {
                "total_nodes": total_nodes,
                "total_edges": total_edges,
                "dataset_field_nodes": dataset_fields,
                "operation_field_nodes": operation_fields,
                "unique_datasets": len(datasets),
                "unique_columns": len(columns),
                "unique_jobs": len(jobs),
                "unique_namespaces": len(namespaces)
            },
            "datasets": sorted(list(datasets)),
            "jobs": sorted(list(jobs))[:20],  # Limit to first 20 for readability
            "namespaces": sorted(list(namespaces)),
            "per_event_graphs_loaded": len(analyzer.per_event_graphs),
            "has_merged_graph": analyzer.merged_graph is not None
        }
        
        return json.dumps(summary, indent=2)
    except Exception as e:
        return f"Error getting lineage summary: {str(e)}"

@mcp.tool()
def find_root_sources(dataset: str, column: Optional[str] = None) -> str:
    """Find all root sources (nodes with no predecessors) for a dataset or column.
    
    Args:
        dataset: Dataset/table name
        column: Optional column name (if not provided, analyzes all columns in dataset)
        
    Returns:
        JSON string with root sources analysis
    """
    try:
        graph = analyzer.merged_graph or nx.compose_all(analyzer.per_event_graphs.values())
        
        # Find target nodes
        target_nodes = analyzer.find_nodes_by_pattern(dataset=dataset, column=column)
        if not target_nodes:
            return f"No nodes found for {dataset}" + (f".{column}" if column else "")
        
        # Trace upstream to find all reachable nodes
        upstream_result = analyzer.trace_lineage(target_nodes, direction="upstream")
        
        # Find root sources (nodes with no predecessors in the reachable set)
        reachable_nodes = set(path["node"] for path in upstream_result["paths"])
        root_sources = []
        
        for node in reachable_nodes:
            predecessors = list(graph.predecessors(node))
            # Check if any predecessor is in our reachable set
            has_reachable_predecessor = any(pred in reachable_nodes for pred in predecessors)
            
            if not has_reachable_predecessor:
                node_data = graph.nodes.get(node, {})
                parsed = node_data.get("parsed", {})
                root_sources.append({
                    "node": node,
                    "parsed": parsed,
                    "runs": node_data.get("runs", []),
                    "jobs": node_data.get("jobs", []),
                    "namespaces": node_data.get("namespaces", []),
                    "resolved_dataset": parsed.get("resolved_dataset"),
                    "resolved_column": parsed.get("resolved_column"),
                    "total_predecessors": len(predecessors)
                })
        
        result = {
            "target": f"{dataset}" + (f".{column}" if column else ""),
            "target_nodes": target_nodes,
            "total_upstream_nodes": len(reachable_nodes),
            "root_sources_count": len(root_sources),
            "root_sources": root_sources
        }
        
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error finding root sources: {str(e)}"

@mcp.tool()
def find_leaf_destinations(dataset: str, column: Optional[str] = None) -> str:
    """Find all leaf destinations (nodes with no successors) for a dataset or column.
    
    Args:
        dataset: Dataset/table name
        column: Optional column name (if not provided, analyzes all columns in dataset)
        
    Returns:
        JSON string with leaf destinations analysis
    """
    try:
        graph = analyzer.merged_graph or nx.compose_all(analyzer.per_event_graphs.values())
        
        # Find source nodes
        source_nodes = analyzer.find_nodes_by_pattern(dataset=dataset, column=column)
        if not source_nodes:
            return f"No nodes found for {dataset}" + (f".{column}" if column else "")
        
        # Trace downstream to find all reachable nodes
        downstream_result = analyzer.trace_lineage(source_nodes, direction="downstream")
        
        # Find leaf destinations (nodes with no successors in the reachable set)
        reachable_nodes = set(path["node"] for path in downstream_result["paths"])
        leaf_destinations = []
        
        for node in reachable_nodes:
            successors = list(graph.successors(node))
            # Check if any successor is in our reachable set
            has_reachable_successor = any(succ in reachable_nodes for succ in successors)
            
            if not has_reachable_successor:
                node_data = graph.nodes.get(node, {})
                parsed = node_data.get("parsed", {})
                leaf_destinations.append({
                    "node": node,
                    "parsed": parsed,
                    "runs": node_data.get("runs", []),
                    "jobs": node_data.get("jobs", []),
                    "namespaces": node_data.get("namespaces", []),
                    "resolved_dataset": parsed.get("resolved_dataset"),
                    "resolved_column": parsed.get("resolved_column"),
                    "total_successors": len(successors)
                })
        
        result = {
            "source": f"{dataset}" + (f".{column}" if column else ""),
            "source_nodes": source_nodes,
            "total_downstream_nodes": len(reachable_nodes),
            "leaf_destinations_count": len(leaf_destinations),
            "leaf_destinations": leaf_destinations
        }
        
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error finding leaf destinations: {str(e)}"

@mcp.tool()
def intelligent_lineage_analysis(query: str, dataset: Optional[str] = None, column: Optional[str] = None, 
                               max_depth: Optional[int] = None) -> str:
    """Intelligent tool that analyzes the user query and calls appropriate lineage analysis functions.
    
    This is the main orchestrator tool that:
    1. Analyzes the user's natural language query
    2. Determines which analysis tools to call
    3. Executes multiple tools in sequence if needed
    4. Combines results into a comprehensive response
    
    Args:
        query: Natural language query describing what lineage analysis is needed
        dataset: Optional dataset/table name if specified
        column: Optional column name if specified  
        max_depth: Optional maximum depth for tracing operations
        
    Returns:
        JSON string with comprehensive analysis results
    """
    try:
        query_lower = query.lower()
        results = {"query": query, "analysis_steps": [], "combined_results": {}}
        
        # Determine analysis type based on query keywords
        analysis_plan = []
        
        # Check for data loading needs first
        if any(word in query_lower for word in ["load", "import", "read", "initialize"]):
            if "merged" in query_lower or "single" in query_lower:
                analysis_plan.append("suggest_load_merged")
            elif "event" in query_lower or "per-event" in query_lower:
                analysis_plan.append("suggest_load_events")
        
        # Impact analysis patterns
        if any(phrase in query_lower for phrase in ["impact", "affect", "change", "modify", "downstream", "consequences"]):
            if dataset and column:
                analysis_plan.append("column_impact")
            elif dataset:
                analysis_plan.append("table_downstream")
            else:
                analysis_plan.append("suggest_specify_target")
        
        # Root cause / upstream analysis
        if any(phrase in query_lower for phrase in ["source", "upstream", "origin", "where.*come", "root", "cause"]):
            if dataset and column:
                analysis_plan.append("column_upstream")
            elif dataset:
                analysis_plan.append("table_upstream")
                if "root" in query_lower or "ultimate" in query_lower:
                    analysis_plan.append("find_roots")
        
        # Path finding
        if any(phrase in query_lower for phrase in ["path", "flow", "from.*to", "between", "connect"]):
            analysis_plan.append("suggest_flow_analysis")
        
        # Search operations
        if any(phrase in query_lower for phrase in ["search", "find.*column", "contains", "pattern", "like"]):
            analysis_plan.append("column_search")
        
        # Schema evolution
        if any(phrase in query_lower for phrase in ["evolution", "change.*time", "history", "over.*time", "schema"]):
            analysis_plan.append("schema_evolution")
        
        # Summary/overview requests  
        if any(phrase in query_lower for phrase in ["summary", "overview", "statistics", "stats", "total"]):
            analysis_plan.append("lineage_summary")
        
        # Leaf/end destinations
        if any(phrase in query_lower for phrase in ["end", "final", "destination", "leaf", "terminal"]):
            analysis_plan.append("find_leaves")
        
        # Execute analysis plan
        for step in analysis_plan:
            step_result = {"step": step, "success": False, "result": None, "error": None}
            
            try:
                if step == "suggest_load_merged":
                    step_result["result"] = "Please first load your data using load_merged_graph tool with the path to your merged graph file."
                
                elif step == "suggest_load_events":
                    step_result["result"] = "Please first load your data using load_per_event_graphs tool with index and directory paths."
                
                elif step == "column_impact" and dataset and column:
                    impact_result = analyzer.analyze_column_impact(dataset, column)
                    step_result["result"] = impact_result
                
                elif step == "table_downstream" and dataset:
                    nodes = analyzer.find_nodes_by_pattern(dataset=dataset)
                    downstream = analyzer.trace_lineage(nodes, direction="downstream", max_depth=max_depth)
                    step_result["result"] = downstream
                
                elif step == "column_upstream" and dataset and column:
                    nodes = analyzer.find_nodes_by_pattern(dataset=dataset, column=column)
                    upstream = analyzer.trace_lineage(nodes, direction="upstream", max_depth=max_depth)
                    step_result["result"] = upstream
                
                elif step == "table_upstream" and dataset:
                    nodes = analyzer.find_nodes_by_pattern(dataset=dataset)
                    upstream = analyzer.trace_lineage(nodes, direction="upstream", max_depth=max_depth)
                    step_result["result"] = upstream
                
                elif step == "find_roots" and dataset:
                    root_result = _find_root_sources_internal(dataset, column)
                    step_result["result"] = root_result
                
                elif step == "find_leaves" and dataset:
                    leaf_result = _find_leaf_destinations_internal(dataset, column)
                    step_result["result"] = leaf_result
                
                elif step == "column_search":
                    if column:
                        search_result = _search_columns_internal(column)
                        step_result["result"] = search_result
                    else:
                        step_result["result"] = "Please specify a column pattern to search for"
                
                elif step == "schema_evolution" and dataset:
                    evolution_result = analyzer.get_table_schema_evolution(dataset)
                    step_result["result"] = evolution_result
                
                elif step == "lineage_summary":
                    summary_result = _get_summary_internal()
                    step_result["result"] = summary_result
                
                elif step == "suggest_flow_analysis":
                    step_result["result"] = "For path analysis, please use find_data_flow_paths tool with source_dataset and target_dataset parameters."
                
                elif step == "suggest_specify_target":
                    step_result["result"] = "Please specify the dataset and optionally column you want to analyze for impact."
                
                step_result["success"] = True
                
            except Exception as e:
                step_result["error"] = str(e)
            
            results["analysis_steps"].append(step_result)
        
        # If no specific analysis was triggered, provide guidance
        if not analysis_plan:
            results["analysis_steps"].append({
                "step": "query_guidance",
                "success": True,
                "result": """I can help with various lineage analyses:
                
• Impact analysis: "What's affected if I change X?"
• Source tracing: "Where does X come from?"
• Path finding: "Show paths from A to B"
• Column search: "Find columns containing Y"
• Schema evolution: "How has X changed over time?"
• Summary: "Give me an overview of my lineage"

Please specify dataset/column names and what type of analysis you need."""
            })
        
        return json.dumps(results, indent=2)
        
    except Exception as e:
        return f"Error in intelligent analysis: {str(e)}"

# Helper functions for internal use
def _find_root_sources_internal(dataset: str, column: Optional[str] = None) -> Dict[str, Any]:
    """Internal helper for finding root sources."""
    graph = analyzer.merged_graph or nx.compose_all(analyzer.per_event_graphs.values())
    target_nodes = analyzer.find_nodes_by_pattern(dataset=dataset, column=column)
    
    if not target_nodes:
        return {"error": f"No nodes found for {dataset}" + (f".{column}" if column else "")}
    
    upstream_result = analyzer.trace_lineage(target_nodes, direction="upstream")
    reachable_nodes = set(path["node"] for path in upstream_result["paths"])
    root_sources = []
    
    for node in reachable_nodes:
        predecessors = list(graph.predecessors(node))
        has_reachable_predecessor = any(pred in reachable_nodes for pred in predecessors)
        
        if not has_reachable_predecessor:
            node_data = graph.nodes.get(node, {})
            parsed = node_data.get("parsed", {})
            root_sources.append({
                "node": node,
                "parsed": parsed,
                "runs": node_data.get("runs", []),
                "jobs": node_data.get("jobs", []),
                "namespaces": node_data.get("namespaces", []),
                "resolved_dataset": parsed.get("resolved_dataset"),
                "resolved_column": parsed.get("resolved_column")
            })
    
    return {
        "target": f"{dataset}" + (f".{column}" if column else ""),
        "root_sources_count": len(root_sources),
        "root_sources": root_sources
    }

def _find_leaf_destinations_internal(dataset: str, column: Optional[str] = None) -> Dict[str, Any]:
    """Internal helper for finding leaf destinations."""
    graph = analyzer.merged_graph or nx.compose_all(analyzer.per_event_graphs.values())
    source_nodes = analyzer.find_nodes_by_pattern(dataset=dataset, column=column)
    
    if not source_nodes:
        return {"error": f"No nodes found for {dataset}" + (f".{column}" if column else "")}
    
    downstream_result = analyzer.trace_lineage(source_nodes, direction="downstream")
    reachable_nodes = set(path["node"] for path in downstream_result["paths"])
    leaf_destinations = []
    
    for node in reachable_nodes:
        successors = list(graph.successors(node))
        has_reachable_successor = any(succ in reachable_nodes for succ in successors)
        
        if not has_reachable_successor:
            node_data = graph.nodes.get(node, {})
            parsed = node_data.get("parsed", {})
            leaf_destinations.append({
                "node": node,
                "parsed": parsed,
                "runs": node_data.get("runs", []),
                "jobs": node_data.get("jobs", []),
                "namespaces": node_data.get("namespaces", []),
                "resolved_dataset": parsed.get("resolved_dataset"),
                "resolved_column": parsed.get("resolved_column")
            })
    
    return {
        "source": f"{dataset}" + (f".{column}" if column else ""),
        "leaf_destinations_count": len(leaf_destinations),
        "leaf_destinations": leaf_destinations
    }

def _search_columns_internal(pattern: str, case_insensitive: bool = True) -> Dict[str, Any]:
    """Internal helper for column search."""
    matches = analyzer.find_nodes_by_pattern(column=pattern, case_insensitive=case_insensitive)
    by_dataset = defaultdict(list)
    graph = analyzer.merged_graph or nx.compose_all(analyzer.per_event_graphs.values())
    
    for node in matches:
        parsed = graph.nodes.get(node, {}).get("parsed", {})
        # Check both direct and resolved datasets/columns
        datasets = [parsed.get("dataset"), parsed.get("resolved_dataset")]
        columns = [parsed.get("column"), parsed.get("resolved_column")]
        
        for dataset in datasets:
            if dataset:
                for column in columns:
                    if column and column not in by_dataset[dataset]:
                        by_dataset[dataset].append(column)
    
    return {
        "search_pattern": pattern,
        "total_matches": len(matches),
        "matches_by_dataset": dict(by_dataset)
    }

def _get_summary_internal() -> Dict[str, Any]:
    """Internal helper for getting lineage summary."""
    graph = analyzer.merged_graph or nx.compose_all(analyzer.per_event_graphs.values())
    
    total_nodes = graph.number_of_nodes()
    total_edges = graph.number_of_edges()
    
    dataset_fields = 0
    operation_fields = 0
    datasets = set()
    columns = set()
    jobs = set()
    namespaces = set()
    
    for node, attrs in graph.nodes(data=True):
        parsed = attrs.get("parsed", {})
        node_kind = parsed.get("kind")
        
        if node_kind == "datasetField":
            dataset_fields += 1
            if parsed.get("dataset"):
                datasets.add(parsed["dataset"])
            if parsed.get("column"):
                columns.add(parsed["column"])
        elif node_kind == "operationField":
            operation_fields += 1
            if parsed.get("resolved_dataset"):
                datasets.add(parsed["resolved_dataset"])
            if parsed.get("resolved_column"):
                columns.add(parsed["resolved_column"])
        
        # Collect job and namespace info
        for job in attrs.get("jobs", []):
            if job:
                jobs.add(job)
        for ns in attrs.get("namespaces", []):
            if ns:
                namespaces.add(ns)
    
    return {
        "graph_statistics": {
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "dataset_field_nodes": dataset_fields,
            "operation_field_nodes": operation_fields,
            "unique_datasets": len(datasets),
            "unique_columns": len(columns),
            "unique_jobs": len(jobs),
            "unique_namespaces": len(namespaces)
        },
        "datasets": sorted(list(datasets)),
        "jobs": sorted(list(jobs))[:20],  # Limit to first 20 for readability
        "namespaces": sorted(list(namespaces)),
        "per_event_graphs_loaded": len(analyzer.per_event_graphs),
        "has_merged_graph": analyzer.merged_graph is not None
    }

@mcp.tool()
def multi_step_lineage_workflow(primary_dataset: str, primary_column: Optional[str] = None,
                               include_impact_analysis: bool = True, 
                               include_root_analysis: bool = True,
                               include_schema_evolution: bool = False,
                               max_depth: Optional[int] = None) -> str:
    """Execute a comprehensive multi-step lineage analysis workflow for a dataset/column.
    
    This tool orchestrates multiple analyses in sequence to provide a complete picture:
    1. Basic information about the target
    2. Upstream lineage (sources)
    3. Root sources (ultimate origins)
    4. Downstream impact analysis
    5. Schema evolution (if requested)
    
    Args:
        primary_dataset: The main dataset to analyze
        primary_column: Optional specific column to focus on
        include_impact_analysis: Whether to include downstream impact analysis
        include_root_analysis: Whether to find ultimate root sources
        include_schema_evolution: Whether to include schema evolution analysis
        max_depth: Maximum depth for lineage tracing
        
    Returns:
        JSON string with comprehensive multi-step analysis results
    """
    try:
        workflow_results = {
            "target": f"{primary_dataset}" + (f".{primary_column}" if primary_column else ""),
            "workflow_steps": [],
            "summary": {}
        }
        
        # Step 1: Verify target exists
        target_nodes = analyzer.find_nodes_by_pattern(dataset=primary_dataset, column=primary_column)
        step1 = {
            "step": 1,
            "name": "target_verification",
            "success": len(target_nodes) > 0,
            "result": {
                "target_nodes_found": len(target_nodes),
                "target_nodes": target_nodes[:5]  # First 5 for brevity
            }
        }
        workflow_results["workflow_steps"].append(step1)
        
        if not target_nodes:
            workflow_results["summary"]["error"] = "Target not found in lineage data"
            return json.dumps(workflow_results, indent=2)
        
        # Step 2: Upstream lineage analysis
        upstream_result = analyzer.trace_lineage(target_nodes, direction="upstream", max_depth=max_depth)
        step2 = {
            "step": 2,
            "name": "upstream_lineage",
            "success": True,
            "result": {
                "total_upstream_nodes": upstream_result["visited_count"],
                "max_depth_reached": upstream_result["max_depth_reached"],
                "upstream_paths_count": len(upstream_result["paths"])
            }
        }
        workflow_results["workflow_steps"].append(step2)
        
        # Step 3: Root sources analysis (if requested)
        if include_root_analysis:
            root_result = _find_root_sources_internal(primary_dataset, primary_column)
            step3 = {
                "step": 3,
                "name": "root_sources_analysis", 
                "success": "error" not in root_result,
                "result": root_result
            }
            workflow_results["workflow_steps"].append(step3)
        
        # Step 4: Impact analysis (if requested)
        if include_impact_analysis:
            if primary_column:
                impact_result = analyzer.analyze_column_impact(primary_dataset, primary_column)
            else:
                # For table-level impact, trace downstream
                downstream_result = analyzer.trace_lineage(target_nodes, direction="downstream", max_depth=max_depth)
                impact_result = {
                    "downstream_analysis": downstream_result,
                    "total_downstream_nodes": downstream_result["visited_count"]
                }
            
            step4 = {
                "step": 4,
                "name": "impact_analysis",
                "success": True,
                "result": impact_result
            }
            workflow_results["workflow_steps"].append(step4)
        
        # Step 5: Schema evolution (if requested)
        if include_schema_evolution:
            evolution_result = analyzer.get_table_schema_evolution(primary_dataset)
            step5 = {
                "step": 5,
                "name": "schema_evolution",
                "success": "error" not in evolution_result,
                "result": evolution_result
            }
            workflow_results["workflow_steps"].append(step5)
        
        # Generate summary
        successful_steps = sum(1 for step in workflow_results["workflow_steps"] if step["success"])
        workflow_results["summary"] = {
            "total_steps_executed": len(workflow_results["workflow_steps"]),
            "successful_steps": successful_steps,
            "target_found": step1["success"],
            "has_upstream_data": step2["result"]["total_upstream_nodes"] > 0,
            "analysis_complete": successful_steps == len(workflow_results["workflow_steps"])
        }
        
        return json.dumps(workflow_results, indent=2)
        
    except Exception as e:
        return f"Error in multi-step workflow: {str(e)}"

if __name__ == "__main__":
    mcp.run()

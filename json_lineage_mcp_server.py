#!/usr/bin/env python3
"""
OpenLineage MCP Server using FastMCP
Provides tools for data lineage tracing with configurable hop limits
"""

import json
import datetime
import collections
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import argparse

from fastmcp import FastMCP

# Global storage for loaded events
_events_cache = {}
_current_json_path = None

#todo: marquez data in json

def load_events(path: str) -> List[Dict]:
    """Load OpenLineage events from JSON file"""
    global _events_cache, _current_json_path
    
    if _current_json_path == path and path in _events_cache:
        return _events_cache[path]
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON file not found: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if isinstance(data, dict) and "events" in data:
        events = data["events"]
    elif isinstance(data, list):
        events = data
    else:
        raise ValueError("Unexpected JSON format; expected list or {'events': [...]}")
    
    _events_cache[path] = events
    _current_json_path = path
    return events

def parse_field_node(node_str: str) -> Dict:
    """Parse field node string into components"""
    parts = node_str.split(":::")
    kind = parts[0] if parts else None
    
    if kind == "datasetField":
        return {
            "kind": "datasetField",
            "namespace": parts[1] if len(parts) > 1 else None,
            "dataset": parts[2] if len(parts) > 2 else None,
            "column": parts[3] if len(parts) > 3 else None,
            "raw": node_str
        }
    elif kind == "operationField":
        return {
            "kind": "operationField",
            "op_uuid": parts[1] if len(parts) > 1 else None,
            "op_id": parts[2] if len(parts) > 2 else None,
            "column": parts[3] if len(parts) > 3 else None,
            "raw": node_str
        }
    else:
        return {
            "kind": "unknown",
            "parts": parts,
            "raw": node_str
        }

def build_column_index(events: List[Dict]) -> Tuple[Dict, Dict]:
    """Build column and dataset indexes"""
    column_to_datasets = collections.defaultdict(set)
    dataset_event_map = collections.defaultdict(list)
    
    for ei, ev in enumerate(events):
        # Index outputs
        for ds in ev.get("outputs", []):
            name = ds.get("name") or ds.get("dataset") or ds.get("physicalName")
            if name:
                dataset_event_map[name].append((
                    ei, 
                    ev.get("eventTime"), 
                    ev.get("run", {}).get("runId")
                ))
        
        # Index column paths
        trans = ev.get("run", {}).get("facets", {}).get("transformations", {})
        for cp in trans.get("column_paths", []):
            origin = cp.get("origin", "")
            dests = cp.get("destination", [])
            
            # Index origin
            if origin and origin.startswith("datasetField:::"):
                p = parse_field_node(origin)
                if p.get("dataset") and p.get("column"):
                    column_to_datasets[p["column"].lower()].add(p["dataset"])
            
            # Index destinations
            for d in dests:
                if d and d.startswith("datasetField:::"):
                    p = parse_field_node(d)
                    if p.get("dataset") and p.get("column"):
                        column_to_datasets[p["column"].lower()].add(p["dataset"])
    
    return column_to_datasets, dataset_event_map

def choose_event_for_dataset(events: List[Dict], dataset_name: str) -> Optional[Tuple]:
    """Find the most recent event for a dataset"""
    candidates = []
    
    for ei, ev in enumerate(events):
        # Check outputs first
        found = False
        for ds in ev.get("outputs", []):
            name = ds.get("name") or ds.get("dataset") or ds.get("physicalName")
            if name == dataset_name:
                candidates.append((ei, ev.get("eventTime"), ev.get("run", {}).get("runId"), True))
                found = True
                break
        
        if found:
            continue
            
        # Fallback: check column_paths
        trans = ev.get("run", {}).get("facets", {}).get("transformations", {})
        for cp in trans.get("column_paths", []):
            origin = cp.get("origin", "")
            dests = cp.get("destination", [])
            if dataset_name in origin or any(dataset_name in d for d in dests):
                candidates.append((ei, ev.get("eventTime"), ev.get("run", {}).get("runId"), False))
                break
    
    if not candidates:
        return None
    
    def parse_time(t):
        if not t:
            return datetime.datetime.min
        try:
            return datetime.datetime.fromisoformat(t.replace("Z", "+00:00"))
        except Exception:
            return datetime.datetime.min
    
    chosen = sorted(candidates, key=lambda x: parse_time(x[1]), reverse=True)[0]
    return chosen

def trace_column_in_event(ev: Dict, chosen_dataset: str, column_name: str, max_hops: Optional[int] = None) -> Tuple[List, List, Optional[str]]:
    """Trace column lineage with optional hop limit"""
    trans = ev.get("run", {}).get("facets", {}).get("transformations", {})
    col_paths = trans.get("column_paths", [])
    table_nodes = trans.get("table_paths", {}).get("nodes", [])
    op_node_map = {n["id"]: n for n in table_nodes}
    
    # Build mappings
    dest_to_origin = collections.defaultdict(list)
    origin_to_dest = collections.defaultdict(list)
    
    for cp in col_paths:
        origin = cp.get("origin")
        dests = cp.get("destination", [])
        origins = origin if isinstance(origin, list) else ([origin] if origin else [])
        
        for d in dests:
            for o in origins:
                dest_to_origin[d].append({"origin": o, "data": cp.get("data", {})})
                origin_to_dest[o].append({"destination": d, "data": cp.get("data", {})})
    
    # Find starting nodes
    starting_nodes = []
    for node in list(dest_to_origin.keys()) + list(origin_to_dest.keys()):
        if not node:
            continue
        if node.startswith("datasetField:::") and chosen_dataset in node and node.endswith(":::" + column_name):
            starting_nodes.append(node)
    
    if not starting_nodes:
        for node in list(dest_to_origin.keys()) + list(origin_to_dest.keys()):
            if node and "datasetField:::" in node and chosen_dataset in node and column_name in node:
                starting_nodes.append(node)
    
    if not starting_nodes:
        return None, None, "Could not find datasetField nodes for the chosen dataset+column"
    
    # DFS with hop limit
    visited = set()
    trace_steps = []
    
    def add_step(s):
        trace_steps.append(s)
    
    def dfs(node, depth=0, hops=0):
        if node in visited:
            return
        if max_hops is not None and hops >= max_hops:
            add_step({
                "type": "hop_limit_reached",
                "node": node,
                "depth": depth,
                "hops": hops,
                "explanation": f"Reached hop limit ({max_hops}) at node {node}"
            })
            return
            
        visited.add(node)
        origins = dest_to_origin.get(node, [])
        
        if not origins:
            parsed = parse_field_node(node)
            if parsed.get("kind") == "datasetField":
                add_step({
                    "type": "read",
                    "node": node,
                    "dataset": parsed.get("dataset"),
                    "column": parsed.get("column"),
                    "depth": depth,
                    "hops": hops,
                    "explanation": f"Read `{parsed.get('column')}` from dataset `{parsed.get('dataset')}` (source)."
                })
            else:
                add_step({
                    "type": "leaf_unknown",
                    "node": node,
                    "depth": depth,
                    "hops": hops,
                    "explanation": f"No origins found for node {node}."
                })
            return
        
        for entry in origins:
            o = entry.get("origin")
            if not o:
                continue
                
            if o.startswith("datasetField:::"):
                po = parse_field_node(o)
                pd = parse_field_node(node)
                add_step({
                    "type": "map_dataset_to_dataset",
                    "from_node": o,
                    "to_node": node,
                    "from_dataset": po.get("dataset"),
                    "from_column": po.get("column"),
                    "to_dataset": pd.get("dataset"),
                    "to_column": pd.get("column"),
                    "depth": depth,
                    "hops": hops,
                    "explanation": f"Column `{pd.get('column')}` is mapped from `{po.get('dataset')}.{po.get('column')}`."
                })
                dfs(o, depth + 1, hops + 1)
                
            elif o.startswith("operationField:::"):
                po = parse_field_node(o)
                opid = po.get("op_id")
                col = po.get("column")
                
                # Find operation info
                op_full_ids = [k for k in op_node_map.keys() if opid in k]
                op_node_info = op_node_map.get(op_full_ids[0]) if op_full_ids else None
                op_label = op_node_info.get("label") if op_node_info else opid
                
                add_step({
                    "type": "operation",
                    "op_id": opid,
                    "op_label": op_label,
                    "op_field": col,
                    "to_node": node,
                    "depth": depth,
                    "hops": hops,
                    "explanation": f"Operation `{opid}` ({op_label}) produced column `{col}`."
                })
                dfs(o, depth + 1, hops + 1)
            else:
                add_step({
                    "type": "unknown_origin",
                    "origin": o,
                    "to_node": node,
                    "depth": depth,
                    "hops": hops,
                    "explanation": f"Found origin `{o}` feeding `{node}` (unknown type)."
                })
                dfs(o, depth + 1, hops + 1)
    
    for s in starting_nodes:
        dfs(s, depth=0, hops=0)
    
    trace_sorted = sorted(trace_steps, key=lambda x: (x.get("depth", 0)))
    return starting_nodes, trace_sorted, None

# Initialize FastMCP server
mcp = FastMCP("OpenLineage Tracer")

@mcp.tool()
def load_openlineage_file(json_path: str) -> str:
    """
    Load OpenLineage JSON file for analysis
    
    Args:
        json_path: Path to the OpenLineage JSON file
        
    Returns:
        Status message with basic statistics
    """
    try:
        events = load_events(json_path)
        column_to_datasets, dataset_event_map = build_column_index(events)
        
        return json.dumps({
            "status": "success",
            "message": f"Loaded {len(events)} events from {json_path}",
            "statistics": {
                "total_events": len(events),
                "unique_columns": len(column_to_datasets),
                "unique_datasets": len(dataset_event_map)
            }
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": str(e)
        }, indent=2)

@mcp.tool()
def find_datasets_with_column(column_name: str, json_path: str = None) -> str:
    """
    Find all datasets containing a specific column
    
    Args:
        column_name: Name of the column to search for (case-insensitive)
        json_path: Optional path to JSON file (uses last loaded if not provided)
        
    Returns:
        JSON with list of datasets containing the column
    """
    try:
        if json_path:
            events = load_events(json_path)
        else:
            if not _current_json_path:
                return json.dumps({
                    "status": "error",
                    "message": "No JSON file loaded. Use load_openlineage_file first or provide json_path."
                })
            events = _events_cache[_current_json_path]
        
        column_to_datasets, dataset_event_map = build_column_index(events)
        column_lower = column_name.lower()
        
        if column_lower not in column_to_datasets:
            return json.dumps({
                "status": "not_found",
                "message": f"No datasets found containing column '{column_name}'",
                "datasets": []
            })
        
        datasets = sorted(list(column_to_datasets[column_lower]))
        dataset_info = []
        
        for ds in datasets:
            event_count = len(dataset_event_map.get(ds, []))
            dataset_info.append({
                "dataset_name": ds,
                "event_count": event_count
            })
        
        return json.dumps({
            "status": "success",
            "column_name": column_name,
            "datasets": dataset_info,
            "total_datasets": len(datasets)
        }, indent=2)
        
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": str(e)
        })

@mcp.tool()
def trace_column_lineage(
    column_name: str, 
    dataset_name: str, 
    max_hops: int = None,
    json_path: str = None,
    include_structured: bool = True
) -> str:
    """
    Trace the upstream lineage of a column in a specific dataset
    
    Args:
        column_name: Name of the column to trace
        dataset_name: Name of the dataset containing the column
        max_hops: Maximum number of hops to trace (unlimited if None)
        json_path: Optional path to JSON file (uses last loaded if not provided)
        include_structured: Whether to include structured trace data
        
    Returns:
        JSON with human-readable lineage and optionally structured data
    """
    try:
        if json_path:
            events = load_events(json_path)
        else:
            if not _current_json_path:
                return json.dumps({
                    "status": "error",
                    "message": "No JSON file loaded. Use load_openlineage_file first or provide json_path."
                })
            events = _events_cache[_current_json_path]
        
        # Find event for dataset
        chosen_event_info = choose_event_for_dataset(events, dataset_name)
        if not chosen_event_info:
            return json.dumps({
                "status": "error",
                "message": f"No event found referencing dataset '{dataset_name}'"
            })
        
        ei, etime, runid, was_output = chosen_event_info
        ev = events[ei]
        
        # Trace lineage
        starting_nodes, trace_sorted, error = trace_column_in_event(
            ev, dataset_name, column_name, max_hops
        )
        
        if error:
            return json.dumps({
                "status": "error",
                "message": error
            })
        
        # Build human-readable lineage
        human_readable = []
        for step in trace_sorted:
            indent = "  " * step.get("depth", 0)
            hop_info = f" [hop {step.get('hops', 0)}]" if max_hops is not None else ""
            human_readable.append(f"{indent}- {step.get('explanation')}{hop_info}")
        
        result = {
            "status": "success",
            "column_name": column_name,
            "dataset_name": dataset_name,
            "max_hops": max_hops,
            "event_info": {
                "event_index": ei,
                "event_time": etime,
                "run_id": runid,
                "dataset_as_output": was_output
            },
            "lineage": {
                "human_readable": human_readable,
                "total_steps": len(trace_sorted)
            }
        }
        
        if include_structured:
            result["lineage"]["structured"] = trace_sorted
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": str(e)
        })

@mcp.tool() 
def get_column_summary(json_path: str = None) -> str:
    """
    Get summary of all columns and datasets in the loaded file
    
    Args:
        json_path: Optional path to JSON file (uses last loaded if not provided)
        
    Returns:
        JSON summary of columns and datasets
    """
    try:
        if json_path:
            events = load_events(json_path)
        else:
            if not _current_json_path:
                return json.dumps({
                    "status": "error", 
                    "message": "No JSON file loaded. Use load_openlineage_file first or provide json_path."
                })
            events = _events_cache[_current_json_path]
        
        column_to_datasets, dataset_event_map = build_column_index(events)
        
        # Build column summary
        column_summary = []
        for col, datasets in column_to_datasets.items():
            column_summary.append({
                "column_name": col,
                "dataset_count": len(datasets),
                "datasets": sorted(list(datasets))
            })
        
        # Sort by dataset count descending
        column_summary.sort(key=lambda x: x["dataset_count"], reverse=True)
        
        # Build dataset summary  
        dataset_summary = []
        for ds, events_info in dataset_event_map.items():
            dataset_summary.append({
                "dataset_name": ds,
                "event_count": len(events_info)
            })
        
        dataset_summary.sort(key=lambda x: x["event_count"], reverse=True)
        
        return json.dumps({
            "status": "success",
            "summary": {
                "total_columns": len(column_to_datasets),
                "total_datasets": len(dataset_event_map),
                "total_events": len(events)
            },
            "columns": column_summary[:50],  # Limit to top 50
            "datasets": dataset_summary[:50]   # Limit to top 50
        }, indent=2)
        
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": str(e)
        })

if __name__ == "__main__":
    # For development/testing
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000, help="Port to run MCP server on")
    parser.add_argument("--host", default="localhost", help="Host to run MCP server on")  
    args = parser.parse_args()
    
    mcp.run(transport="stdio")

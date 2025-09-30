import json
from collections import defaultdict

def build_graph(marquez_json):
    """Build adjacency map from Marquez JSON edges"""
    edges = marquez_json.get("lineage", {}).get("edges", [])
    graph = defaultdict(list)
    for edge in edges:
        src = edge.get("source")
        tgt = edge.get("target")
        # Transformation info (if available)
        transformation = edge.get("data", {}).get("transformation", "direct mapping")
        if src and tgt:
            graph[src].append((tgt, transformation))
    return graph

def get_node_metadata(marquez_json):
    """Map node_id -> {namespace, table, column, description}"""
    metadata = {}
    nodes = marquez_json.get("lineage", {}).get("nodes", [])
    for node in nodes:
        if node.get("type") == "DATASET":
            namespace = node["data"].get("namespace")
            table = node["data"].get("name")
            for field in node["data"].get("fields", []):
                col_name = field["name"]
                desc = field.get("description") or None
                node_id = f"datasetField::{namespace}::{table}::{col_name}"
                metadata[node_id] = {
                    "namespace": namespace,
                    "table": table,
                    "column": col_name,
                    "description": desc
                }
    return metadata

def build_recursive(node_id, graph, metadata, visited=None):
    """Recursively build lineage tree"""
    if visited is None:
        visited = set()
    if node_id in visited:  # prevent cycles
        return None
    visited.add(node_id)

    node_meta = metadata.get(node_id, {})
    node = {
        "namespace": node_meta.get("namespace"),
        "table": node_meta.get("table"),
        "column": node_meta.get("column"),
        "description": node_meta.get("description"),
        "downstream": []
    }

    downstream_nodes = graph.get(node_id, [])
    for tgt_id, transformation in downstream_nodes:
        child = build_recursive(tgt_id, graph, metadata, visited)
        if child:
            child["transformation"] = transformation
            node["downstream"].append(child)

    return node

def collect_summary(tree):
    """Traverse recursive tree to collect summary info"""
    final_impacts = []
    intermediate_nodes = []
    transformation_types = set()
    governance_gaps = []
    max_depth = 0

    def dfs(node, depth):
        nonlocal max_depth
        if not node.get("downstream"):
            final_impacts.append(f"{node['namespace']}.{node['table']}.{node['column']}")
        else:
            intermediate_nodes.append(f"{node['namespace']}.{node['table']}.{node['column']}")
        if node.get("transformation"):
            transformation_types.add(node["transformation"])
        if not node.get("description"):
            governance_gaps.append(f"{node['namespace']}.{node['table']}.{node['column']}")
        max_depth = max(max_depth, depth)
        for child in node.get("downstream", []):
            dfs(child, depth + 1)

    dfs(tree, 1)
    return {
        "final_impacts": final_impacts,
        "hops": max_depth - 1,
        "intermediate_nodes": intermediate_nodes,
        "transformation_types": list(transformation_types),
        "governance_gaps": governance_gaps
    }

def transform_to_compact(marquez_json, query_column_id):
    graph = build_graph(marquez_json)
    metadata = get_node_metadata(marquez_json)

    # Build recursive lineage tree
    tree = build_recursive(query_column_id, graph, metadata)

    # Build final compact JSON
    compact_json = {
        "query": metadata.get(query_column_id, {}),
        "lineage": tree,
        "summary": collect_summary(tree) if tree else {}
    }
    return compact_json


# Example usage:
if __name__ == "__main__":
    with open("marquez_json_trf.txt", "r") as f:
        marquez_json = json.load(f)

    # Example query column (adjust to match your Marquez node IDs)
    query_column_id = "datasetField:::multi_entity_capital_enriched:::capital_rrds_fact:::c08c005"

    compact = transform_to_compact(marquez_json, query_column_id)

    # Save compact JSON
    with open("compact_lineage.json", "w") as f:
        json.dump(compact, f, indent=2)

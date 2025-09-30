import json
import re
from collections import defaultdict

# -------------------------------
# 1. Load the Marquez JSON
# -------------------------------
input_path = "/mnt/data/marquez_json_trf.txt"
with open(input_path, "r") as f:
    raw = json.load(f)

nodes = {n["id"]: n for n in raw["lineage"]["nodes"]}
edges = raw["lineage"]["edges"]

# Build adjacency maps
out_map = defaultdict(list)
for e in edges:
    out_map[e["origin"]].append(e["destination"])


# -------------------------------
# 2. Utilities
# -------------------------------
def get_fields(dataset_id):
    """Return set of fields for a dataset node."""
    node = nodes.get(dataset_id)
    if not node:
        return set()
    return {f["name"] for f in node["data"].get("fields", [])}


def classify_field(src_col, src_dataset, op_node, field):
    """
    Classify how 'field' in operation output relates to src_col in src_dataset.
    """
    src_fields = get_fields(src_dataset)

    if field == src_col:
        return "direct"
    elif field in src_fields:
        return "direct_other"
    elif re.match(r"^(sum_|avg_|count_|min_|max_)", field, re.I):
        return "aggregation"
    else:
        return f"rename_or_derived_from_{src_col}"


# -------------------------------
# 3. DFS Traversal
# -------------------------------
def dfs(dataset_id, col, visited):
    """
    Recursive traversal: dataset.col -> downstream
    """
    key = f"{dataset_id}.{col}"
    if key in visited:
        return None
    visited.add(key)

    dataset_node = nodes.get(dataset_id)
    if not dataset_node:
        return None

    lineage_node = {
        "namespace": dataset_node["data"]["namespace"],
        "table": dataset_node["data"]["name"],
        "column": col,
        "description": None,
        "transformation": None,
        "downstream": []
    }

    # find description if available
    for f in dataset_node["data"].get("fields", []):
        if f["name"] == col:
            lineage_node["description"] = f.get("description")

    # walk downstream
    for nxt in out_map.get(dataset_id, []):
        nxt_node = nodes.get(nxt)
        if not nxt_node:
            continue

        if nxt_node["type"] == "OPERATION":
            transf = f"{nxt_node['data'].get('type')}: {nxt_node['data'].get('label')}"
            for f in nxt_node["data"].get("fields", []):
                mtype = classify_field(col, dataset_id, nxt_node, f)
                for nxt2 in out_map.get(nxt, []):
                    nxt2_node = nodes.get(nxt2)
                    if nxt2_node and nxt2_node["type"] == "DATASET":
                        child = dfs(nxt2, f, visited.copy())
                        if child:
                            child["transformation"] = f"{mtype} via {transf}"
                            lineage_node["downstream"].append(child)

    return lineage_node


# -------------------------------
# 4. Summarize
# -------------------------------
def collect_summary(node, depth=0, summary=None):
    if summary is None:
        summary = {
            "final_impacts": [],
            "hops": 0,
            "intermediate_nodes": set(),
            "transformation_types": set(),
            "governance_gaps": []
        }

    if not node["downstream"]:
        summary["final_impacts"].append(
            f"{node['namespace']}.{node['table']}.{node['column']}"
        )

    summary["hops"] = max(summary["hops"], depth)

    if node["transformation"]:
        ttype = node["transformation"].split()[0]
        summary["transformation_types"].add(ttype)

    if not node.get("description") or node["description"] == "No description available.":
        summary["governance_gaps"].append(
            f"{node['namespace']}.{node['table']}.{node['column']}"
        )

    for child in node["downstream"]:
        summary["intermediate_nodes"].add(f"{child['namespace']}.{child['table']}")
        collect_summary(child, depth + 1, summary)

    return summary


# -------------------------------
# 5. Run for specific column
# -------------------------------
query_dataset = "dataset:::multi_entity_capital_enriched:::capital_rrds_fact"
query_col = "c08c005"

tree = dfs(query_dataset, query_col, visited=set())
summary = collect_summary(tree)

compact_json = {
    "query": {
        "namespace": nodes[query_dataset]["data"]["namespace"],
        "table": nodes[query_dataset]["data"]["name"],
        "column": query_col,
        "description": next(
            (f["description"] for f in nodes[query_dataset]["data"]["fields"] if f["name"] == query_col),
            None
        )
    },
    "lineage": tree,
    "summary": {
        "final_impacts": summary["final_impacts"],
        "hops": summary["hops"],
        "intermediate_nodes": list(summary["intermediate_nodes"]),
        "transformation_types": list(summary["transformation_types"]),
        "governance_gaps": summary["governance_gaps"]
    }
}

# -------------------------------
# 6. Save Output
# -------------------------------
output_path = "/mnt/data/compact_lineage_c08c005.json"
with open(output_path, "w") as f:
    json.dump(compact_json, f, indent=2)

print(f"✅ Compact JSON written to {output_path}")

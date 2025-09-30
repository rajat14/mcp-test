import json
from collections import defaultdict
from typing import Dict, Set, Tuple

def ds_key(ds_id: str) -> Tuple[str, str]:
    parts = ds_id.split(":::")
    return (parts[1], parts[2]) if len(parts) >= 3 else (ds_id, "")

def load_marquez(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_maps(data):
    lineage = data.get("lineage", {}) or {}
    nodes = lineage.get("nodes", []) or []
    edges = lineage.get("edges", []) or []

    dataset_nodes: Dict[str, dict] = {}
    dataset_fields: Dict[str, Dict[str, str]] = {}
    operation_nodes: Dict[str, dict] = {}

    for n in nodes:
        nid = n.get("id")
        ntype = n.get("type")
        if ntype == "DATASET":
            dataset_nodes[nid] = n
            fields_map = {}
            for f in (n.get("data") or {}).get("fields", []) or []:
                if f and isinstance(f, dict):
                    name = f.get("name")
                    if name:
                        fields_map[name] = f.get("description")
            dataset_fields[nid] = fields_map
        elif ntype == "OPERATION":
            operation_nodes[nid] = n

    # Collapse dataset→operation→dataset
    op_inputs: Dict[str, Set[str]] = defaultdict(set)   # op_id -> set(dataset_id)
    op_outputs: Dict[str, Set[str]] = defaultdict(set)  # op_id -> set(dataset_id)

    for e in edges:
        origin = e.get("origin")
        dest = e.get("destination")
        if origin in dataset_nodes and dest in operation_nodes:
            op_inputs[dest].add(origin)
        if origin in operation_nodes and dest in dataset_nodes:
            op_outputs[origin].add(dest)

    # Operation fields & short transformation label
    op_fields: Dict[str, Set[str]] = {}
    op_label: Dict[str, str] = {}
    for op_id, op in operation_nodes.items():
        od = (op.get("data") or {})
        flist = od.get("fields") or []
        op_fields[op_id] = set(f for f in flist if isinstance(f, str))
        otype = od.get("type")
        label = od.get("label")
        op_label[op_id] = f"{otype}: {label}" if otype and label else (otype or label or "unknown")

    return dataset_nodes, dataset_fields, operation_nodes, op_inputs, op_outputs, op_fields, op_label

def build_column_graph(dataset_fields, op_inputs, op_outputs, op_fields, op_label):
    """
    Build a column-level downstream mapping using a heuristic:
    - If an operation touches a field name (in op.fields) present in an input dataset,
      and the same field name is present in an output dataset, map input.col -> output.col (same name).
    - If names differ but both appear in op.fields and exist in respective datasets, map input.col -> a few output cols.
    """
    from collections import defaultdict
    downstream = defaultdict(list)  # (ds_id, col) -> [((out_ds_id, out_col), transformation)]
    MAX_CROSS_MAPS = 5

    for op_id, inputs in op_inputs.items():
        outs = op_outputs.get(op_id, set())
        if not outs:
            continue
        fields_in_op = op_fields.get(op_id, set())
        trans = op_label.get(op_id, "unknown")

        for in_ds in inputs:
            in_cols = dataset_fields.get(in_ds, {})
            in_overlap = fields_in_op & set(in_cols.keys())
            if not in_overlap:
                continue
            for out_ds in outs:
                out_cols = dataset_fields.get(out_ds, {})
                out_overlap = fields_in_op & set(out_cols.keys())
                if not out_overlap:
                    continue

                same = in_overlap & out_overlap
                # 1) Same-name mappings (strong signal)
                for c in sorted(same):
                    downstream[(in_ds, c)].append(((out_ds, c), trans))

                # 2) Cross-name mappings (weak signal; capped)
                targets = sorted(list(out_overlap))
                for src_c in sorted(in_overlap - same):
                    for tgt_c in targets[:MAX_CROSS_MAPS]:
                        downstream[(in_ds, src_c)].append(((out_ds, tgt_c), trans))
    return downstream

def build_tree(start_ds_id, start_col, dataset_fields, downstream):
    visited = set()
    max_depth = 0

    def rec(ds_id, col, depth=1):
        nonlocal max_depth
        max_depth = max(max_depth, depth)
        key = (ds_id, col)
        if key in visited:
            return None
        visited.add(key)

        ns, tbl = ds_key(ds_id)
        node = {
            "namespace": ns,
            "table": tbl,
            "column": col,
            "description": dataset_fields.get(ds_id, {}).get(col)
        }
        children = []
        for (next_ds, next_col), trans in downstream.get(key, []):
            child = rec(next_ds, next_col, depth + 1)
            if child:
                child["transformation"] = trans
                children.append(child)
        if children:
            node["downstream"] = children
        return node

    return rec(start_ds_id, start_col), max_depth

def summarize(tree, max_depth):
    final_impacts, intermediate_nodes, transformation_types, governance_gaps = [], [], set(), []

    def dfs(n):
        if not n:
            return
        pid = f"{n['namespace']}.{n['table']}.{n['column']}"
        if n.get("downstream"):
            intermediate_nodes.append(pid)
        else:
            final_impacts.append(pid)
        if n.get("transformation"):
            t = n["transformation"]
            transformation_types.add(t.split(":")[0].strip() if ":" in t else t)
        if not n.get("description"):
            governance_gaps.append(pid)
        for ch in n.get("downstream", []):
            dfs(ch)

    if tree:
        dfs(tree)
    return {
        "final_impacts": final_impacts,
        "hops": max_depth - 1 if tree else 0,
        "intermediate_nodes": intermediate_nodes,
        "transformation_types": sorted(transformation_types),
        "governance_gaps": governance_gaps
    }

def transform(marquez_path: str, namespace: str, table: str, column: str, out_path: str):
    data = load_marquez(marquez_path)
    ds_nodes, ds_fields, op_nodes, op_inputs, op_outputs, op_fields, op_label = build_maps(data)

    # resolve dataset id
    ns_tbl_to_id = {}
    for ds_id in ds_nodes:
        ns, tbl = ds_key(ds_id)
        ns_tbl_to_id[f"{ns}.{tbl}"] = ds_id
    start_ds_id = ns_tbl_to_id.get(f"{namespace}.{table}")
    if not start_ds_id:
        raise ValueError(f"Dataset not found: {namespace}.{table}")

    downstream = build_column_graph(ds_fields, op_inputs, op_outputs, op_fields, op_label)
    tree, max_depth = build_tree(start_ds_id, column, ds_fields, downstream)

    compact = {
        "query": {
            "namespace": namespace,
            "table": table,
            "column": column,
            "description": ds_fields.get(start_ds_id, {}).get(column)
        },
        "lineage": tree,
        "summary": summarize(tree, max_depth)
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(compact, f, indent=2)

# ---- Example run on your uploaded file ----
if __name__ == "__main__":
    transform(
        marquez_path="/mnt/data/marquez_json_trf.txt",
        namespace="multi_entity_capital_enriched",
        table="capital_rrds_fact",
        column="c08c005",
        out_path="/mnt/data/compact_lineage.json",
    )

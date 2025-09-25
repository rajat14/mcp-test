def filter_lineage_json(lineage_file_path, query_column=None, query_namespace=None, query_table=None, direction="downstream"):
    """
    Filter lineage JSON to extract only essential results for a specific column lineage query.
    
    Args:
        lineage_file_path (str): Path to the JSON file containing lineage data
        query_column (str): Name of the column to trace (e.g., 'c08c005')
        query_namespace (str): Namespace of the source table (e.g., 'multi_entity_capital_enriched')
        query_table (str): Name of the source table (e.g., 'capital_rrds_fact')
        direction (str): 'downstream', 'upstream', or 'both'
    
    Returns:
        dict: Minimal filtered lineage data with just the essential results
    """
    import json
    
    # Load lineage data from file
    try:
        with open(lineage_file_path, 'r') as file:
            lineage_data = json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Lineage file not found: {lineage_file_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in file: {lineage_file_path}")
    
    if not lineage_data or 'lineage' not in lineage_data:
        return lineage_data
    
    nodes = lineage_data['lineage'].get('nodes', [])
    edges = lineage_data['lineage'].get('edges', [])
    
    # If no specific query parameters, return original (for general queries)
    if not query_column:
        return lineage_data
    
    # Create lookup dictionaries for faster access
    node_lookup = {node['id']: node for node in nodes}
    
    # Find the source field ID
    source_field_id = f"datasetField:::{query_namespace}:::{query_table}:::{query_column}"
    
    # Results containers
    downstream_columns = []
    transformation_jobs = []
    upstream_columns = []
    
    def extract_table_info(field_id):
        """Extract namespace, table, and column from field ID"""
        parts = field_id.split(':::')
        if len(parts) >= 4 and parts[0] == 'datasetField':
            return {
                'namespace': parts[1],
                'table': parts[2], 
                'column': parts[3]
            }
        return None
    
    def find_downstream_lineage(start_field_id, visited=None):
        """Find all downstream columns from a field"""
        if visited is None:
            visited = set()
        
        if start_field_id in visited:
            return
        
        visited.add(start_field_id)
        
        # Find edges where this field is in the 'previous' list
        for edge in edges:
            edge_data = edge.get('data', {})
            previous_fields = edge_data.get('previous', [])
            next_fields = edge_data.get('next', [])
            
            if start_field_id in previous_fields:
                # Add transformation job
                destination = edge['destination']
                if destination.startswith('job:::'):
                    job_node = node_lookup.get(destination)
                    if job_node:
                        job_name = job_node['data'].get('name', destination.split(':::')[-1])
                        if job_name not in transformation_jobs:
                            transformation_jobs.append(job_name)
                
                # Add downstream columns
                for next_field in next_fields:
                    field_info = extract_table_info(next_field)
                    if field_info and field_info not in downstream_columns:
                        downstream_columns.append(field_info)
                    
                    # Continue recursively
                    find_downstream_lineage(next_field, visited)
    
    def find_upstream_lineage(target_field_id, visited=None):
        """Find all upstream columns to a field"""
        if visited is None:
            visited = set()
        
        if target_field_id in visited:
            return
        
        visited.add(target_field_id)
        
        # Find edges where this field is in the 'next' list
        for edge in edges:
            edge_data = edge.get('data', {})
            previous_fields = edge_data.get('previous', [])
            next_fields = edge_data.get('next', [])
            
            if target_field_id in next_fields:
                # Add transformation job
                destination = edge['destination']
                if destination.startswith('job:::'):
                    job_node = node_lookup.get(destination)
                    if job_node:
                        job_name = job_node['data'].get('name', destination.split(':::')[-1])
                        if job_name not in transformation_jobs:
                            transformation_jobs.append(job_name)
                
                # Add upstream columns
                for prev_field in previous_fields:
                    field_info = extract_table_info(prev_field)
                    if field_info and field_info not in upstream_columns:
                        upstream_columns.append(field_info)
                    
                    # Continue recursively
                    find_upstream_lineage(prev_field, visited)
    
    # Execute tracing based on direction
    if direction in ['downstream', 'both']:
        find_downstream_lineage(source_field_id)
    
    if direction in ['upstream', 'both']:
        find_upstream_lineage(source_field_id)
    
    # Build minimal result structure
    result = {
        "source": {
            "column": query_column,
            "table": query_table,
            "namespace": query_namespace
        }
    }
    
    if transformation_jobs:
        result["transformation_jobs"] = transformation_jobs
    
    if direction in ['downstream', 'both'] and downstream_columns:
        result["downstream_columns"] = downstream_columns
    
    if direction in ['upstream', 'both'] and upstream_columns:
        result["upstream_columns"] = upstream_columns
    
    return result


def extract_query_params(user_query):
    """
    Extract column, namespace, and table from natural language query.
    
    Args:
        user_query (str): User's natural language query
    
    Returns:
        tuple: (column, namespace, table, direction)
    """
    import re
    
    query_lower = user_query.lower()
    
    # Determine direction
    direction = "downstream"
    if "upstream" in query_lower or "source" in query_lower or "origin" in query_lower:
        direction = "upstream"
    elif "both" in query_lower or "full" in query_lower:
        direction = "both"
    
    column = None
    namespace = None
    table = None
    
    # Look for patterns like "column:x", "namespace:y", "table:z"
    column_match = re.search(r'column:([a-zA-Z0-9_]+)', user_query)
    if column_match:
        column = column_match.group(1)
    
    # Look for patterns like "namespace:y"
    namespace_match = re.search(r'namespace:([a-zA-Z0-9_]+)', user_query)
    if namespace_match:
        namespace = namespace_match.group(1)
    
    # Look for patterns like "table:z"
    table_match = re.search(r'table:([a-zA-Z0-9_]+)', user_query)
    if table_match:
        table = table_match.group(1)
    
    return column, namespace, table, direction


def process_lineage_query(lineage_file_path, user_query, output_file_path=None):
    """
    Main function to process a lineage query and return minimal filtered results.
    
    Args:
        lineage_file_path (str): Path to the JSON file containing lineage data
        user_query (str): User's natural language query
        output_file_path (str, optional): Path to save the filtered results
    
    Returns:
        dict: Minimal filtered lineage data ready for LLM summarization
    """
    
    # Extract query parameters
    column, namespace, table, direction = extract_query_params(user_query)
    
    # Filter the lineage data
    filtered_data = filter_lineage_json(
        lineage_file_path, 
        query_column=column,
        query_namespace=namespace, 
        query_table=table,
        direction=direction
    )
    
    # Add query context
    final_result = {
        "query": user_query,
        "extracted_params": {
            "column": column,
            "namespace": namespace,
            "table": table,
            "direction": direction
        },
        "lineage_result": filtered_data
    }
    
    # Save to file if output path is provided
    if output_file_path:
        import json
        with open(output_file_path, 'w') as file:
            json.dump(final_result, file, indent=2)
    
    return final_result

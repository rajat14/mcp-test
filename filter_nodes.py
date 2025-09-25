def filter_lineage_json(lineage_data, query_column=None, query_namespace=None, query_table=None, direction="downstream"):
    """
    Filter lineage JSON to extract only relevant information for a specific column lineage query.
    
    Args:
        lineage_data (dict): The full lineage JSON response
        query_column (str): Name of the column to trace (e.g., 'c08c005')
        query_namespace (str): Namespace of the source table (e.g., 'multi_entity_capital_enriched')
        query_table (str): Name of the source table (e.g., 'capital_rrds_fact')
        direction (str): 'downstream', 'upstream', or 'both'
    
    Returns:
        dict: Filtered lineage data containing only relevant nodes and edges
    """
    
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
    
    # Set to store all relevant node IDs
    relevant_nodes = set()
    relevant_edges = []
    
    def add_related_dataset_and_job(field_id):
        """Add the parent dataset and any related jobs for a field"""
        parts = field_id.split(':::')
        if len(parts) >= 3 and parts[0] == 'datasetField':
            # Add parent dataset
            dataset_id = f"dataset:::{parts[1]}:::{parts[2]}"
            relevant_nodes.add(dataset_id)
    
    def trace_downstream(start_field_id, visited=None):
        """Recursively trace downstream lineage from a field"""
        if visited is None:
            visited = set()
        
        if start_field_id in visited:
            return
        
        visited.add(start_field_id)
        relevant_nodes.add(start_field_id)
        add_related_dataset_and_job(start_field_id)
        
        # Find edges where this field is in the 'previous' list
        for edge in edges:
            edge_data = edge.get('data', {})
            previous_fields = edge_data.get('previous', [])
            next_fields = edge_data.get('next', [])
            
            if start_field_id in previous_fields:
                # Add the job/transformation
                destination = edge['destination']
                relevant_nodes.add(destination)
                relevant_edges.append(edge)
                
                # Continue tracing to next fields
                for next_field in next_fields:
                    trace_downstream(next_field, visited)
    
    def trace_upstream(target_field_id, visited=None):
        """Recursively trace upstream lineage to a field"""
        if visited is None:
            visited = set()
        
        if target_field_id in visited:
            return
        
        visited.add(target_field_id)
        relevant_nodes.add(target_field_id)
        add_related_dataset_and_job(target_field_id)
        
        # Find edges where this field is in the 'next' list
        for edge in edges:
            edge_data = edge.get('data', {})
            previous_fields = edge_data.get('previous', [])
            next_fields = edge_data.get('next', [])
            
            if target_field_id in next_fields:
                # Add the job/transformation
                origin = edge['origin']
                relevant_nodes.add(origin)
                relevant_edges.append(edge)
                
                # Continue tracing to previous fields
                for prev_field in previous_fields:
                    trace_upstream(prev_field, visited)
    
    # Start tracing based on direction
    if direction in ['downstream', 'both']:
        trace_downstream(source_field_id)
    
    if direction in ['upstream', 'both']:
        trace_upstream(source_field_id)
    
    # Add dataset-to-job and job-to-dataset edges for completeness
    for edge in edges:
        origin = edge['origin']
        destination = edge['destination']
        
        if origin in relevant_nodes or destination in relevant_nodes:
            if origin.startswith('dataset:::') or destination.startswith('dataset:::') or \
               origin.startswith('job:::') or destination.startswith('job:::'):
                relevant_edges.append(edge)
                relevant_nodes.add(origin)
                relevant_nodes.add(destination)
    
    # Filter nodes and edges
    filtered_nodes = [node for node in nodes if node['id'] in relevant_nodes]
    
    # Remove duplicate edges
    unique_edges = []
    edge_signatures = set()
    for edge in relevant_edges:
        signature = (edge['origin'], edge['destination'])
        if signature not in edge_signatures:
            unique_edges.append(edge)
            edge_signatures.add(signature)
    
    # Create filtered response
    filtered_data = {
        'lineage': {
            'nodes': filtered_nodes,
            'edges': unique_edges
        }
    }
    
    return filtered_data


def extract_query_params(user_query):
    """
    Extract column, namespace, and table from natural language query.
    This is a simple implementation - you might want to use more sophisticated NLP.
    
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
    
    # Extract parameters using regex or keyword matching
    # This is a simplified approach - adapt based on your query patterns
    
    column = None
    namespace = None
    table = None
    
    # Look for patterns like "column x", "field x"
    column_match = re.search(r'(?:column|field)\s+(\w+)', query_lower)
    if column_match:
        column = column_match.group(1)
    
    # Look for patterns like "namespace y"
    namespace_match = re.search(r'namespace\s+([a-zA-Z0-9_]+)', query_lower)
    if namespace_match:
        namespace = namespace_match.group(1)
    
    # Look for patterns like "table z"
    table_match = re.search(r'table\s+([a-zA-Z0-9_]+)', query_lower)
    if table_match:
        table = table_match.group(1)
    
    return column, namespace, table, direction


# Example usage function
def process_lineage_query(lineage_json, user_query):
    """
    Main function to process a lineage query and return filtered results.
    
    Args:
        lineage_json (dict): Full lineage response from API
        user_query (str): User's natural language query
    
    Returns:
        dict: Filtered lineage data ready for LLM summarization
    """
    
    # Extract query parameters
    column, namespace, table, direction = extract_query_params(user_query)
    
    # Filter the lineage data
    filtered_data = filter_lineage_json(
        lineage_json, 
        query_column=column,
        query_namespace=namespace, 
        query_table=table,
        direction=direction
    )
    
    # Add metadata about the filtering
    filtered_data['query_info'] = {
        'original_query': user_query,
        'extracted_params': {
            'column': column,
            'namespace': namespace,
            'table': table,
            'direction': direction
        },
        'nodes_filtered': len(lineage_json.get('lineage', {}).get('nodes', [])),
        'edges_filtered': len(lineage_json.get('lineage', {}).get('edges', [])),
        'nodes_remaining': len(filtered_data.get('lineage', {}).get('nodes', [])),
        'edges_remaining': len(filtered_data.get('lineage', {}).get('edges', []))
    }
    
    return filtered_data

import requests
from fastmcp import FastMCP

# Base URL for Marquez API (Docker default is 5000)
MARQUEZ_API = "http://localhost:5000/api/v1"
DEFAULT_NAMESPACE = "food_delivery"

# Create FastMCP server
mcp = FastMCP("marquez-server")

@mcp.tool
def get_namespaces():
    """Fetch all namespaces from Marquez."""
    url = f"{MARQUEZ_API}/namespaces"
    resp = requests.get(url)
    if resp.status_code != 200:
        return {"error": f"Failed to fetch namespaces: {resp.text}"}
    return resp.json()

@mcp.tool
def get_datasets(namespace: str = DEFAULT_NAMESPACE):
    """Fetch all datasets in a given namespace (default: food_delivery)."""
    url = f"{MARQUEZ_API}/namespaces/{namespace}/datasets"
    resp = requests.get(url)
    if resp.status_code != 200:
        return {"error": f"Failed to fetch datasets: {resp.text}"}
    return resp.json()

@mcp.tool
def get_jobs(namespace: str = DEFAULT_NAMESPACE):
    """Fetch all jobs in a given namespace (default: food_delivery)."""
    url = f"{MARQUEZ_API}/namespaces/{namespace}/jobs"
    resp = requests.get(url)
    if resp.status_code != 200:
        return {"error": f"Failed to fetch jobs: {resp.text}"}
    return resp.json()

@mcp.tool
def get_lineage(dataset: str, namespace: str = DEFAULT_NAMESPACE, depth: int = 2):
    """
    Fetch lineage for a dataset in Marquez (default namespace: food_delivery).

    Args:
        dataset: The dataset name (e.g., 'orders', 'deliveries').
        namespace: The namespace (default: food_delivery).
        depth: How many hops to traverse in the lineage graph.
    """
    url = f"{MARQUEZ_API}/lineage?nodeId={dataset}:namespace={namespace}&depth={depth}"
    resp = requests.get(url)
    if resp.status_code != 200:
        return {"error": f"Failed to fetch lineage: {resp.text}"}
    return resp.json()

if __name__ == "__main__":
    mcp.run()

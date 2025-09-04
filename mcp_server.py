from mcp.server.fastmcp import FastMCP
import httpx
import json
from typing import Tuple, Dict
import requests
# from metatool import mcp
from pydantic import BaseModel

# Initialize FastMCP with async support
mcp = FastMCP("Lineage Application", enable_async=True)

# Define input schema for FastAPI to parse JSON body
class SmartDataRetrievalInput(BaseModel):
    user_query: str



@mcp.tool()
async def smart_data_retrieval(input: SmartDataRetrievalInput) -> str:
    """
    This tool retrieves data based on user query by calling the smart data retrieval API.
    The API returns  four different variables related to the table and column in the user query:
    - Column names
    - Table names
    - File types
    - Column descriptions
    Args:
        user_query: Natural language query about database table and columns and their information
        use this function to view available table and column information.
    Returns:
        Formatted JSON response containing column information
    """
    async with httpx.AsyncClient() as client:
        try:
            headers = {
                "Content-Type": "application/json"
            }
            url = "http://localhost:8000/api/smart_data_retrieval"
            payload = {"user_query": input.user_query}
            response = await client.post(  
                url,
                headers=headers,
                json=payload,
                timeout=120.0
            )
            response.raise_for_status()
            # Return the JSON response as a formatted string
            return json.dumps(response.json(), indent=2)
        except httpx.HTTPStatusError as e:
            return f"HTTP error occurred: {e.response.status_code} - {e.response.text}"
        except httpx.RequestError as e:
            return f"Request error occurred: {str(e)}"
        except Exception as e:
            return f"Unexpected error: {str(e)}"

@mcp.tool()
async def trace_lineage(  table_name: str,
            column_name: str) -> str:           #Tuple[str, Dict, str, Dict]
    """
    Based on the information available from "smart_data_retrieval" tool, fetch the relevant table
     name and column name from the "smart_data_retrieval" tool response.
    Then, use this function to trace lineage of a specific column in a table.
    Args:
        table_name: Name of the table selected from the "smart_data_retrieval" tool response
        column_name: Name of the column to trace lineage from the "smart_data_retrieval" tool response
    Returns:
        Formatted JSON response containing column information
    """
    async with httpx.AsyncClient() as client:
        try:
            headers = {
                "Content-Type": "application/json"
            }
            url = "http://localhost:8000/api/trace_lineage"

            payload = {
        
                "table_name": table_name,
                "column_name": column_name
            }

            headers = {
                "Content-Type": "application/json"
            }

            #response = requests.post(url, json=payload, headers=headers)
            response = await client.post(url, json=payload, headers=headers, timeout=120.0)
            response.raise_for_status()

            # Return the JSON response as a formatted string
            return json.dumps(response.json(), indent=2)
        except httpx.HTTPStatusError as e:
            return f"HTTP error occurred: {e.response.status_code} - {e.response.text}"
        except httpx.RequestError as e:
            return f"Request error occurred: {str(e)}"
        except Exception as e:
            return f"Unexpected error: {str(e)}"

# If you want to run the MCP server
if __name__ == "__main__":
    mcp.run()

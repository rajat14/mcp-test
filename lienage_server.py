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

class RequestModel(BaseModel):
    input: SmartDataRetrievalInput

class TextToSqlInput(BaseModel):
    user_query: str

class ExecuteSqlInput(BaseModel):
    sql_query: str



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
            url = "http://localhost:8000/api/v1/smart_data_retrieval"
            payload = {"user_query": input.user_query}
            response = await client.post(  # Changed to GET request
                url,
                headers=headers,
                json=payload.model_dump_json(),
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
async def text_to_sql(input: TextToSqlInput) -> str:
    """
    Convert a natural language query into an SQL statement.
    Args:
        user_query: Question in plain English about the database.
    Returns:
        JSON containing the generated SQL query.
    """
    async with httpx.AsyncClient() as client:
        try:
            headers = {
                "Content-Type": "application/json"
            }
            url = "http://localhost:8000/api/v1/text_to_sql"
            payload = {"user_query": input.user_query}
            response = await client.post(
                url,
                headers=headers,
                json=payload,
                timeout=120.0
            )
            response.raise_for_status()
            # response = await client.post(url, json=input.dict(), timeout=120.0)
            # response.raise_for_status()
            return json.dumps(response.json(), indent=2)
        except httpx.HTTPStatusError as e:
            return f"HTTP error: {e.response.status_code} - {e.response.text}"
        except httpx.RequestError as e:
            return f"Request error: {str(e)}"
        except Exception as e:
            return f"Unexpected error: {str(e)}"


@mcp.tool()
async def execute_sql(input: ExecuteSqlInput) -> str:
    """
    Execute a generated SQL query and return the result.
    Args:
        sql_query: SQL query string to run on the database.
    Returns:
        JSON with the execution result.
    """
    async with httpx.AsyncClient() as client:
        try:
            headers = {
                "Content-Type": "application/json"
            }
            url = "http://localhost:8000/api/v1/text_to_sql/execute"
            payload = {"sql_query": input.sql_query}
            response = await client.post(
                url,
                headers=headers,
                json=payload,
                timeout=120.0
            )
            response.raise_for_status()
            # url = "http://localhost:8000/api/v1/text_to_sql/execute"
            # response = await client.post(url, json=input.dict(), timeout=120.0)
            # response.raise_for_status()
            return json.dumps(response.json(), indent=2)
        except httpx.HTTPStatusError as e:
            return f"HTTP error: {e.response.status_code} - {e.response.text}"
        except httpx.RequestError as e:
            return f"Request error: {str(e)}"
        except Exception as e:
            return f"Unexpected error: {str(e)}"

# If you want to run the MCP server
if __name__ == "__main__":
    mcp.run()

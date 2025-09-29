from pydantic import BaseModel

class McpExecuteRequest(BaseModel):
    user_query: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "user_query": "Find all tables with a column named 'customer_id' and provide a summary of their usage."
                }
            ]
        }
    }

class McpToolResponse(BaseModel):
    tools: list

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "tools": []
                }
            ]
        }
    }

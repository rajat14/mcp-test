
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager
from mcp_client import MCPClient
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
import traceback
import markdown
import asyncio

load_dotenv()
class Settings(BaseSettings):
    use_server_manager: bool = True
    # Store the entire MCP servers config
    mcp_servers: dict = {
        "serper": {
            "command": "python",
            "args": [".\\mcp-client-python-example-master\\serper_mcp_server\\server.py"],
            "env": {
                "SERPER_API_KEY": "51ad0802af8cec50f90a68a75b0005e9ba64d4be"
            }
        },
        "sequential-thinking": {
            "command": "npx",
            "args": ["-y","@modelcontextprotocol/server-sequential-thinking"]
            
    },
        "lineage":{
            "command": "python",
            "args": [".\\testing_mcp\\lienage_server.py"]
    
},
        "marquez": {
            "command": "python",
            "args": [".\\mcp-client-python-example-master\\api\\marquez_server.py"],
            "env": {}
}

#         "sqlite":{
#             "command": "python",
#             "args": [".\\testing_mcp\\sqlite_mcp_server.py"]
    
# }
    }
    #Specify which server to use
    active_server: str = "lineage"
    
settings = Settings()
## MODIFIED ## - Globals for both modes
app_client: Optional[MCPClient] = None # For single-server mode
current_active_server: str = settings.active_server # For single-server mode

## ADDED ## - Globals for Server Manager mode
active_connections: Dict[str, MCPClient] = {}
tool_to_server_map: Dict[str, str] = {}
all_tools: List[dict] = []

async def switch_server(new_server_name: str):
    """Switch to a different server by recreating the client connection"""
    global app_client, current_active_server
    
    if new_server_name not in settings.mcp_servers:
        raise HTTPException(status_code=404, detail=f"Server '{new_server_name}' not found in configuration")
    
    old_server = current_active_server

    # Track server switch before clean up existing client
    if app_client:
        app_client.session_tracker.track_server_switch(old_server, new_server_name)
        await app_client.cleanup()
    
    # Create new client and connect to new server
    app_client = MCPClient()
    server_config = settings.mcp_servers[new_server_name]
    connected = await app_client.connect_to_server(server_config)
    
    if not connected:
        raise HTTPException(status_code=500, detail=f"Failed to connect to server '{new_server_name}'")
    
    current_active_server = new_server_name
    return app_client

@asynccontextmanager
async def lifespan(app: FastAPI):
    global app_client, current_active_server, active_connections, tool_to_server_map, all_tools
    if settings.use_server_manager:
        #ADDED ## - Server Manager Startup Logic
        print("Starting in Server Manager mode...")
        for server_name, server_config in settings.mcp_servers.items():
            print(f"Connecting to server: {server_name}")
            client = MCPClient()
            try:
                await client.connect_to_server(server_config)
                active_connections[server_name] = client
                for tool in client.tools:
                    if tool['name'] in tool_to_server_map:
                        print(f"Warning: Duplicate tool name '{tool['name']}' found. Overwriting.")
                    tool_to_server_map[tool['name']] = server_name
                all_tools.extend(client.tools)
                print(f"Successfully connected to {server_name}. Tools found: {[t['name'] for t in client.tools]}")
            except Exception as e:
                print(f"Failed to connect to server {server_name}: {e}")
        print("Server Manager startup complete.")
        print(f"Full tool-to-server map: {tool_to_server_map}")
        yield
        # Shutdown
        print("Shutting down all server connections concurrently...")
        await asyncio.gather(
            *(client.cleanup() for client in active_connections.values())
        )
        print("All connections closed.")
    else:    
        print(f"Starting in single-server mode with active server: {settings.active_server}")
        app_client = MCPClient()
    #client = MCPClient()
        try:
            # Get the specific server config
            server_config = settings.mcp_servers[settings.active_server]
            #connected = await client.connect_to_server(server_config)
            connected = await app_client.connect_to_server(server_config)
            if not connected:
                raise Exception("Failed to connect to default server: '{settings.active_server}'")
            #app.state.client = client
            app.state.client = app_client
            yield
        except Exception as e:
            raise Exception(f"Failed to connect to server: {str(e)}")
        finally:
            # Shutdown
            if app_client:
            #await client.cleanup()
                await app_client.cleanup()
app = FastAPI(title="MCP Chatbot API", lifespan=lifespan)
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
class QueryRequest(BaseModel):
    query: str
    server_name: Optional[str] = None
    selectedItem: Optional[Dict[str, Any]] = None
class SwitchServerRequest(BaseModel):
    server_name: str
class Message(BaseModel):
    role: str
    content: Any
class ToolCall(BaseModel):
    name: str
    args: Dict[str, Any]
    #server_name: Optional[str] = None

@app.get("/servers")
async def list_available_servers():
    """Get list of available servers and current active server"""
    return {
        "server_manager_enabled": settings.use_server_manager,
        "available_servers": list(settings.mcp_servers.keys()),
        #"connected_servers": list(active_connections.keys())
        "active_server": list(active_connections.keys()) if settings.use_server_manager else current_active_server
    }

@app.post("/switch-server")
async def switch_active_server(request: SwitchServerRequest):
    """Switch to a different server"""
    try:
        await switch_server(request.server_name)
        return {
            "message": f"Successfully switched to server '{request.server_name}'",
            "active_server": current_active_server
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/session-metrics")
async def get_session_metrics():
    try:
        global app_client, active_connections
        if settings.use_server_manager:
            if not active_connections:
                raise HTTPException(status_code=400, detail="No active server connections in Server Manager mode")
            
            # Aggregate all metrics into one summary
            combined = {
                "queries": 0,
                "tool_calls": 0,
                "server_switches": 0,
                "total_tokens": 0,
                "estimated_cost_usd": 0.0,
                "errors": 0,
                "top_tools": {}
            }
            
            for client in active_connections.values():
                m = client.get_session_metrics()
                combined["queries"] += m["queries"]
                combined["tool_calls"] += m["tool_calls"]
                combined["server_switches"] += m["server_switches"]
                combined["total_tokens"] += m["total_tokens"]
                combined["estimated_cost_usd"] += m["estimated_cost_usd"]
                combined["errors"] += m["errors"]
                for tool, count in m["top_tools"]:
                    combined["top_tools"][tool] = combined["top_tools"].get(tool, 0) + count

            # convert top_tools dict to sorted list of tuples
            top_tools_sorted = sorted(combined["top_tools"].items(), key=lambda x: x[1], reverse=True)[:5]
            combined["top_tools"] = top_tools_sorted
            
            return {"metrics": combined}
        
        else:
            if app_client:
                return {"metrics": app_client.get_session_metrics()}
            else:
                raise HTTPException(status_code=400, detail="No active client session")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tools")
async def get_available_tools():
    """Get list of available tools from current active server"""
    if settings.use_server_manager:
        return {
            "server": "manager",
            "tools": all_tools
        }
    else:
        try:
            global app_client
            tools = await app_client.get_mcp_tools()
            return {
                "server": current_active_server,
                "tools": [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.inputSchema,
                    }
                    for tool in tools
                ]
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def process_query(request: QueryRequest):
    """Process a query and return the response"""
    if settings.use_server_manager:
        # ----- Server Manager branch -----
        if not active_connections:
            raise HTTPException(status_code=500, detail="Server Manager is enabled, but no servers are connected.")
        
        # Use any client for the LLM call, conversation history shared
        temp_client = next(iter(active_connections.values()))
        messages_history = [{"role": "user", "content": request.query}]
        temp_client.messages = messages_history.copy()
        temp_client.session_tracker.track_query(request.query, "manager")
        try:
            while True:
                if request.selectedItem:
                    context = f"User selected this item for explanation:\n{json.dumps(request.selectedItem, indent=2)}"
                    response = await temp_client.call_llm(f"{context}\nPlease explain this clearly.")
                    return {"response":response.to_dict()["content"]}
                response = await temp_client.call_llm(tools_override=all_tools)
                if response.content[0].type == "text" and len(response.content) == 1:
                    assistant_message = {"role": "assistant", "content": markdown.markdown(response.content[0].text)}
                    messages_history.append(assistant_message)
                    break
                assistant_message = {"role": "assistant", "content": response.to_dict()["content"]}
                messages_history.append(assistant_message)
                temp_client.messages.append(assistant_message)

                for content in response.content:
                    if content.type == "tool_use":
                        tool_name = content.name
                        tool_args = content.input
                        tool_use_id = content.id
                        server_name = tool_to_server_map.get(tool_name)
                        if not server_name:
                            raise Exception(f"Tool '{tool_name}' not found in any connected server.")
                        tool_client = active_connections[server_name]
                        result = await tool_client.call_tool(tool_name, tool_args)
                        # ADD THIS:
                        tool_client.session_tracker.track_tool_call(tool_name, server_name)
                        tool_result_message = {
                            "role": "user",
                            "content": [{"type": "tool_result", "tool_use_id": tool_use_id, "content": result.content}]
                        }
                        messages_history.append(tool_result_message)
                        temp_client.messages.append(tool_result_message)

            return {"server_used": "manager", "messages": messages_history}
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

    else:
        # ----- Single Server branch -----
        try:
            global app_client, current_active_server
            if request.server_name and request.server_name != current_active_server:
                from main import switch_server
                await switch_server(request.server_name)

            messages = await app_client.process_query(request.query, current_active_server)
            return {
                "server_used": current_active_server,
                "messages": messages
            }
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/tool")
async def call_tool(tool_call: ToolCall):
    """Call a specific tool on the current active server"""
    try:
        global app_client
        # Track tool call
        app_client.session_tracker.track_tool_call(tool_call.name, current_active_server)
        
        result = await app_client.call_tool(tool_call.name, tool_call.args)
        return {
            "server_used": current_active_server,
            "result": result
        }
    except Exception as e:
        if app_client:
            app_client.session_tracker.track_error("API_TOOL_ERROR", str(e))
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008)

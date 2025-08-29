
from typing import Optional, List
from contextlib import AsyncExitStack
import traceback
from utils.logger import logger
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from datetime import datetime
import json
import os
import markdown
from anthropic import Anthropic
from anthropic import AnthropicVertex
from anthropic.types import Message
from dotenv import load_dotenv
# Add this import at the top of mcp_client.py
from session_tracker import SessionTracker
# Load .env file
load_dotenv()
# Get the credential path from .env
#credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'GCP-Service-Account-Key.json'

# Initialize client
client = AnthropicVertex(
    region="xyz",  # or your preferred region
    project_id="abc"
)



class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        # self.llm = Anthropic()
        self.llm = client
        self.tools = []
        self.messages = []
        self.logger = logger
        self.session_tracker = SessionTracker()
    async def call_tool(self, tool_name: str, tool_args: dict):
        """Call a tool with the given name and arguments"""
        try:
            result = await self.session.call_tool(tool_name, tool_args)
            return result
        except Exception as e:
            self.logger.error(f"Failed to call tool: {str(e)}")
            raise Exception(f"Failed to call tool: {str(e)}")
    async def connect_to_server(self, server_config: dict):
        """Connect to an MCP server
        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        try:
            command = server_config["command"]
            args = server_config["args"] 
            env = server_config.get("env", {})
            server_params = StdioServerParameters(
                command=command, 
                args=args, 
                env=env
            )
            self.logger.info(f"Running command: {command} {' '.join(args)}")
            self.logger.info(f"With environment: {env}")
            # server_params = StdioServerParameters(
            #     command=command, args=[server_script_path], env=None
            # )
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            self.stdio, self.write = stdio_transport
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(self.stdio, self.write)
            )
            await self.session.initialize()
            mcp_tools = await self.get_mcp_tools()
            self.tools = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema,
                }
                for tool in mcp_tools
            ]
            self.logger.info(
                f"Successfully connected to server. Available tools: {[tool['name'] for tool in self.tools]}"
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to server: {str(e)}")
            self.logger.debug(f"Connection error details: {traceback.format_exc()}")
            raise Exception(f"Failed to connect to server: {str(e)}")
    async def get_mcp_tools(self):
        try:
            self.logger.info("Requesting MCP tools from the server.")
            response = await self.session.list_tools()
            tools = response.tools
            return tools
        except Exception as e:
            self.logger.error(f"Failed to get MCP tools: {str(e)}")
            self.logger.debug(f"Error details: {traceback.format_exc()}")
            raise Exception(f"Failed to get tools: {str(e)}")

    async def call_llm(self, tools_override: Optional[List] = None) -> Message:
        """Call the LLM with the given query"""
        try:
            active_tools = tools_override if tools_override is not None else self.tools
            response = self.llm.messages.create(
                model='claude-sonnet',
                max_tokens=4000,
                messages=self.messages,
                tools=active_tools,
            )
        
            # Track token usage
            input_tokens = response.usage.input_tokens if hasattr(response, 'usage') else 0
            output_tokens = response.usage.output_tokens if hasattr(response, 'usage') else 0
            self.session_tracker.track_llm_usage(input_tokens, output_tokens, "claude-sonnet-4")
        
            return response
        except Exception as e:
            self.session_tracker.track_error("LLM_CALL_ERROR", str(e))
            self.logger.error(f"Failed to call LLM: {str(e)}")
            raise Exception(f"Failed to call LLM: {str(e)}")
        try:
            return self.llm.messages.create(
                #model="claude-3-5-sonnet-20241022",
                model = 'claude-sonnet-4',
                max_tokens=4000,
                messages=self.messages,
                tools=self.tools,
            )
        
        except Exception as e:
            self.logger.error(f"Failed to call LLM: {str(e)}")
            raise Exception(f"Failed to call LLM: {str(e)}")
    async def process_query(self, query: str, server_name : str  = "default"):
        """Process a query using Claude and available tools, returning all messages at the end"""
        try:
            self.session_tracker.track_query(query, server_name)
            self.logger.info(
                f"Processing new query: {query[:100]}..."
            )  # Log first 100 chars of query
            # Add the initial user message
            user_message = {"role": "user", "content": query}
            self.messages.append(user_message)
            await self.log_conversation(self.messages)
            messages = [user_message]
            while True:
                self.logger.debug("Calling Claude API")
                response = await self.call_llm()
                # If it's a simple text response
                if response.content[0].type == "text" and len(response.content) == 1:
                    assistant_message = {
                        "role": "assistant",
                        "content": markdown.markdown(response.content[0].text),
                    }
                    self.messages.append(assistant_message)
                    await self.log_conversation(self.messages)
                    messages.append(assistant_message)
                    break
                # For more complex responses with tool calls
                assistant_message = {
                    "role": "assistant",
                    "content": response.to_dict()["content"],
                }
                self.messages.append(assistant_message)
                await self.log_conversation(self.messages)
                messages.append(assistant_message)
                for content in response.content:
                    if content.type == "text":
                        # Text content within a complex response
                        text_message = {"role": "assistant", "content": markdown.markdown(content.text)}
                        await self.log_conversation(self.messages)
                        messages.append(text_message)
                    elif content.type == "tool_use":
                        tool_name = content.name
                        tool_args = content.input
                        tool_use_id = content.id
                        self.logger.info(
                            f"Executing tool: {tool_name} with args: {tool_args}"
                        )
                        # Track tool call
                        self.session_tracker.track_tool_call(tool_name, server_name)
                        try:
                            # turn this one return a simple string
                            result = await self.session.call_tool(tool_name, tool_args)
                            self.logger.info(f"Tool result: {result}")
                            # result = test_tool_result_content
                            tool_result_message = {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "tool_result",
                                        "tool_use_id": tool_use_id,
                                        "content": result.content,
                                    }
                                ],
                            }
                            self.messages.append(tool_result_message)
                            await self.log_conversation(self.messages)
                            messages.append(tool_result_message)
                        except Exception as e:
                            error_msg = f"Tool execution failed: {str(e)}"
                            self.logger.error(error_msg)
                            raise Exception(error_msg)
            return messages
        except Exception as e:
            self.session_tracker.track_error("QUERY_PROCESSING_ERROR", str(e))
            self.logger.error(f"Error processing query: {str(e)}")
            self.logger.debug(
                f"Query processing error details: {traceback.format_exc()}"
            )
            raise
    # Add a method to get session metrics:
    def get_session_metrics(self):
        """Get current session metrics"""
        return self.session_tracker.get_metrics_summary()
    async def log_conversation(self, conversation: list):
        """Log the conversation to json file"""
        # Create conversations directory if it doesn't exist
        os.makedirs("conversations", exist_ok=True)
        # Convert conversation to JSON-serializable format
        serializable_conversation = []
        for message in conversation:
            try:
                serializable_message = {
                    "role": message["role"],
                    "content": []
                }
                
                # Handle both string and list content
                if isinstance(message["content"], str):
                    serializable_message["content"] = message["content"]                  
                elif isinstance(message["content"], list):
                    for content_item in message["content"]:
                        if hasattr(content_item, 'to_dict'):
                            serializable_message["content"].append(content_item.to_dict())
                        elif hasattr(content_item, 'dict'):
                            serializable_message["content"].append(content_item.dict())
                        elif hasattr(content_item, 'model_dump'):
                            serializable_message["content"].append(content_item.model_dump())
                        else:
                            serializable_message["content"].append(content_item)
                
                serializable_conversation.append(serializable_message)
            except Exception as e:
                self.logger.error(f"Error processing message: {str(e)}")
                self.logger.debug(f"Message content: {message}")
                raise
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filepath = os.path.join("conversations", f"conversation_{timestamp}.json")
        
        try:
            with open(filepath, "w") as f:
                json.dump(serializable_conversation, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Error writing conversation to file: {str(e)}")
            self.logger.debug(f"Serializable conversation: {serializable_conversation}")
            raise
    async def cleanup(self):
        """Clean up resources"""
        try:
            self.session_tracker.finalize_session()
            self.logger.info("Cleaning up resources")
            await self.exit_stack.aclose()
        except Exception as e:
            self.session_tracker.track_error("CLEANUP_ERROR", str(e))
            self.logger.error(f"Error during cleanup: {str(e)}")


import asyncio
import langchain_mcp_adapters
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
# from langchain_openai import ChatOpenAI
from langchain_openai import AzureChatOpenAI
import os
from dotenv import load_dotenv


async def main():

    # Load environment variables
    load_dotenv(dotenv_path = r".\mcp-client-python-example-master\api\.env")

    # 1. Define your MCP servers (stdio)
    mcp_servers = {
        "serper": {
            "command": "python",
            "args": ["./serper_mcp_server/server.py"], 
            "transport": "stdio"
        },
        "marquez": {
            "command": "python",
            "args": ["./api/marquez_server.py"],
            "transport": "stdio"
        }
    }

    # 2. Connect to MCP servers
    client = MultiServerMCPClient(mcp_servers)
    tools = client.get_tools()

    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")

    

    # 4. Define agents with restricted toolsets
    data_expert_agent = AzureChatOpenAI(
    deployment_name="abc",
    api_key=api_key,
    azure_endpoint=endpoint,
    temperature=0.3,
    api_version=api_version
)
    web_search_agent = AzureChatOpenAI(
    deployment_name="abc",
    api_key=api_key,
    azure_endpoint=endpoint,
    temperature=0.3,
    api_version=api_version
)

     # 4) Create LangGraph agents with scoped toolsets
    serper_tools = [t for t in tools if "serper" in t.name.lower()]
    marquez_tools = [t for t in tools if "marquez" in t.name.lower()]

    graph_web = create_react_agent(web_search_agent, tools=serper_tools) 
    graph_data = create_react_agent(data_expert_agent, tools=marquez_tools) 

     # 5) Example queries (messages accepted as list or plain string)
    resp1 = await graph_data.ainvoke(
        {"messages": [{"role": "user", "content": "Show lineage for orders table"}]}
    )  # ainvoke accepts {"messages": ...} per docs [4][5]
    print("Data Expert Response:", resp1)

    resp2 = await graph_web.ainvoke(
        {"messages": [{"role": "user", "content": "Search latest AI breakthroughs"}]}
    )  # also supports {"messages": "your prompt"} shorthand [8][4]
    print("Web Search Response:", resp2)


if __name__ == "__main__":
    asyncio.run(main())

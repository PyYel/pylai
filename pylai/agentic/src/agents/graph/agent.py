from typing import Any
from crewai import Agent
from crewai.mcp import MCPServerHTTP

from tools.graph.tools import GraphQueryTool, GraphGetNeighborsTool
from pylcloud.database import DatabaseGraphLocal


class GraphAgent(Agent):
    """Overload a CrewAI agent with both custom tools and MCP Server capabilities"""

    def __init__(
        self,
        db_host: str,
        db_user: str,
        db_password: str,
        mcp_urls: list[str] = [],
        mcp_tokens: list[str] = [],
        **kwargs: Any,
    ) -> None:

        self.client = DatabaseGraphLocal(host=db_host)
        self.client.connect_database(host=db_host)

        # Custom internal tools
        tools = [
            GraphQueryTool(graph_db_interface=self.client),
            GraphGetNeighborsTool(graph_db_interface=self.client),
        ]

        mcps = []
        for mcp_url, mcp_token in zip(mcp_urls, mcp_tokens):
            graph_mcp = MCPServerHTTP(
                url=mcp_url,
                headers={"Authorization": f"Bearer {mcp_token}"},
                cache_tools_list=True,  # Optimization to prevent polling the server every single turn
            )
            mcps.append(graph_mcp)

        super().__init__(
            role="Graph browsing agent",
            goal="Find matching entities and relationships from a knowledge graph.",
            backstory="An expert of <topic>",
            tools=tools,
            mcps=mcps,
            **kwargs,
        )

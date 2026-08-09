from typing import Any
from crewai import Agent
from crewai.mcp import MCPServerHTTP

from tools.rag.tools import RAGSearchTool, RAGIngestTool


class RAGAgent(Agent):
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

        self.client = VectorDBInterface(
            db_host=db_host, db_user=db_user, db_password=db_password
        )

        # Custom internal tools
        tools = [
            RAGSearchTool(vector_db_interface=self.client),
            RAGIngestTool(vector_db_interface=self.client),
        ]

        mcps = []
        for mcp_url, mcp_token in zip(mcp_urls, mcp_tokens):
            rag_mcp = MCPServerHTTP(
                url=mcp_url,
                headers={"Authorization": f"Bearer {mcp_token}"},
                cache_tools_list=True,  # Optimization to prevent polling the server every single turn
            )
            mcps.append(rag_mcp)

        super().__init__(
            role="RAG browsing agent",
            goal="Find matching sources from a knowledge base.",
            backstory="An expert of <topic>",
            tools=tools,
            mcps=mcps,
            **kwargs,
        )

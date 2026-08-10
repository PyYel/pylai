"""
pylai.agentic utility helpers: check an MCP server before wiring it into an
agent, and spin up a simple local interactive chatbot for any agent.
"""

import asyncio
from typing import Any
from pydantic_ai.mcp import MCPToolset

from .agents.agent import Agent


def test_mcp_server_sync(url: str, token: str = ""):
    """Sync wrapper around ``test_mcp_server`` — run it directly from a script or REPL."""

    async def test_mcp_server(url: str, token: str = ""):
        """Connects to an MCP server and prints the tools it exposes.

        A quick sanity check for an endpoint before wiring it into an agent via
        ``mcp_urls=``/``mcp_tokens=``.

        Args:
            url: The MCP server's fully qualified endpoint (e.g. "https://example.com/mcp").
            token: Optional bearer token for server-side auth.
        """
        toolset = MCPToolset(
            url,
            headers={"Authorization": f"Bearer {token}"} if token else None,
        )

        tools = []
        print(f"Connecting to MCP server at {url}...")
        try:
            async with toolset:
                tools = await toolset.list_tools()
                print(f"\n--- {len(tools)} tool(s) available ---")
                for tool in tools:
                    print(f"- {tool.name}: {tool.description}")
        except Exception as e:
            print(f"\n--- Connection failed ---\n{type(e).__name__}: {e}")
        return [tool.model_dump() for tool in tools]

    return asyncio.run(test_mcp_server(url, token))


def start_local_chatbot(agent: Agent, **cli_kwargs: Any) -> None:
    """Starts a simple interactive CLI chat loop for an agent.

    Parameters
    ----------
    agent: Agent
        The agent to chat with.
    **cli_kwargs: Any
        Forwarded to ``agent.to_cli_sync`` (e.g. ``message_history=``).
    """
    agent.to_cli_sync(prog_name=agent.name or "pylai-agent", **cli_kwargs)

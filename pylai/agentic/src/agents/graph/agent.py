from typing import Any, Sequence

from pydantic_ai import Agent
from pydantic_ai.models import KnownModelName, Model

from pylcloud.database.src.graph.DatabaseGraph import DatabaseGraph

from .._instructions import build_mcp_toolsets, role_instructions
from ...tools.graph.tools import build_graph_neighbors_tool, build_graph_query_tool


class GraphAgent(Agent):
    """
    Ready-to-use knowledge-graph agent: connect an already-connected pylcloud
    graph database, get a working graph-browsing agent.
    """

    DEFAULT_ROLE = "Graph browsing agent"
    DEFAULT_GOAL = "Find matching entities and relationships from a knowledge graph."
    DEFAULT_BACKSTORY = "An expert at traversing and explaining knowledge-graph structure."

    def __init__(
        self,
        db_client: DatabaseGraph,
        *,
        model: Model | KnownModelName | str,
        mcp_urls: Sequence[str] = (),
        mcp_tokens: Sequence[str] = (),
        role: str = DEFAULT_ROLE,
        goal: str = DEFAULT_GOAL,
        backstory: str = DEFAULT_BACKSTORY,
        **agent_kwargs: Any,
    ) -> None:
        self.client = db_client

        tools = [
            build_graph_query_tool(db_client),
            build_graph_neighbors_tool(db_client),
        ]
        toolsets = build_mcp_toolsets(mcp_urls, mcp_tokens)

        super().__init__(
            model=model,
            name="graph_agent",
            description=goal,
            instructions=role_instructions(role, goal, backstory),
            tools=tools,
            toolsets=toolsets or None,
            **agent_kwargs,
        )

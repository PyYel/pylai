from typing import Any, Optional, Sequence, Union

from pydantic_ai.models import KnownModelName, Model

from pylcloud.database.src.graph.DatabaseGraph import DatabaseGraph

from ..agent import Agent
from ...tools.graph.tools import GraphNeighborsTool, GraphQueryTool

class GraphAgent(Agent):
    """
    Ready-to-use knowledge-graph agent: connect an already-connected pylcloud
    graph database, get a working graph-browsing agent.
    """

    DEFAULT_INSTRUCTIONS = (
        "Answer questions using the knowledge graph. Query it for matching nodes "
        "and edges before answering. Use node ids to query, but answer the user "
        "using the entities' actual names/labels when available, and expand a "
        "node's neighbors when the question is about relationships."
    )
    DEFAULT_DESCRIPTION = "Finds and answers questions from a knowledge graph."

    def __init__(
        self,
        db_client: DatabaseGraph,
        model: Union[Model, KnownModelName],
        name: str = "GraphAgent",
        mcp_urls: Sequence[str] = (),
        mcp_tokens: Sequence[str] = (),
        instructions: Optional[str] = None,
        description: Optional[str] = None,
        **agent_kwargs: Any,
    ) -> None:
        """
        Instanciates a ``GraphAgent``.

        This agent has some ready to use tools to read and browse a graph structure or database,
        depending on the ``db_client`` type passed. 
        
        Parameters
        ----------
        model: str
            The LLM model tied to this agent execution.

        Notes
        -----
        This agent and its tools are built upon the database module of the [PyYelCloud](https://github.com/PyYel/pylcloud) library:
        ```
        pip install pylcloud[database]
        ```
        """
        self.client = db_client

        tools = [
            GraphQueryTool(db_client),
            GraphNeighborsTool(db_client),
        ]

        super().__init__(
            model=model,
            name=name,
            tools=tools,
            mcp_urls=mcp_urls,
            mcp_tokens=mcp_tokens,
            instructions=instructions,
            description=description,
            **agent_kwargs,
        )

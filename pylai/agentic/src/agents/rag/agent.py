from typing import Any, Optional, Sequence, Union

from pydantic_ai.models import KnownModelName, Model

from pylcloud.database.src.search.DatabaseSearch import DatabaseSearch
from pylcloud.gpt.src.GPT import GPT

from ..agent import Agent
from ...tools.rag.tools import RAGIngestTool, RAGSearchTool


class RAGAgent(Agent):
    """
    Ready-to-use RAG agent: connect a pylcloud vector store + embedder, get a
    working knowledge-base agent. Both must already be constructed and connected
    by the caller — RAGAgent does not assume which search backend
    (Elasticsearch/Opensearch/S3Vector) or embedding provider you're using.
    """

    DEFAULT_INSTRUCTIONS = (
        "Answer questions using the knowledge base. Search it for chunks relevant "
        "to the question before answering, and ground your answer in what you "
        "find rather than guessing. Ingest new content into the knowledge base "
        "when asked to remember or store something."
    )
    DEFAULT_DESCRIPTION = "Finds and answers questions from a knowledge base."

    def __init__(
        self,
        db_client: DatabaseSearch,
        embedder: GPT,
        index_name: str,
        embedding_model_name: str,
        model: Union[Model, KnownModelName],
        name: str = "RAGAgent",
        mcp_urls: Sequence[str] = (),
        mcp_tokens: Sequence[str] = (),
        instructions: Optional[str] = None,
        description: Optional[str] = None,
        **agent_kwargs: Any,
    ) -> None:
        """
        Instanciates a ``RAGAgent``.

        This agent has some ready to use tools to read and browse a vector database,
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
        self.embedder = embedder

        tools = [
            RAGSearchTool(db_client, embedder, embedding_model_name, index_name),
            RAGIngestTool(db_client, embedder, embedding_model_name, index_name),
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

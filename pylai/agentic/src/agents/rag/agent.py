from typing import Any, Sequence

from pydantic_ai import Agent
from pydantic_ai.models import KnownModelName, Model

from pylcloud.database.src.search.DatabaseSearch import DatabaseSearch
from pylcloud.gpt.src.GPT import GPT

from .._instructions import build_mcp_toolsets, role_instructions
from ...tools.rag.tools import build_rag_ingest_tool, build_rag_search_tool


class RAGAgent(Agent):
    """
    Ready-to-use RAG agent: connect a pylcloud vector store + embedder, get a
    working knowledge-base agent. Both must already be constructed and connected
    by the caller — RAGAgent does not assume which search backend
    (Elasticsearch/Opensearch/S3Vector) or embedding provider you're using.
    """

    DEFAULT_ROLE = "RAG browsing agent"
    DEFAULT_GOAL = "Find matching sources from a knowledge base and answer questions grounded in them."
    DEFAULT_BACKSTORY = "An expert at retrieving and synthesizing knowledge-base content."

    def __init__(
        self,
        db_client: DatabaseSearch,
        embedder: GPT,
        *,
        index_name: str,
        embedding_model_name: str,
        model: Model | KnownModelName | str,
        mcp_urls: Sequence[str] = (),
        mcp_tokens: Sequence[str] = (),
        role: str = DEFAULT_ROLE,
        goal: str = DEFAULT_GOAL,
        backstory: str = DEFAULT_BACKSTORY,
        **agent_kwargs: Any,
    ) -> None:
        self.client = db_client
        self.embedder = embedder

        tools = [
            build_rag_search_tool(db_client, embedder, embedding_model_name, index_name),
            build_rag_ingest_tool(db_client, embedder, embedding_model_name, index_name),
        ]
        toolsets = build_mcp_toolsets(mcp_urls, mcp_tokens)

        super().__init__(
            model=model,
            name="rag_agent",
            description=goal,
            instructions=role_instructions(role, goal, backstory),
            tools=tools,
            toolsets=toolsets or None,
            **agent_kwargs,
        )

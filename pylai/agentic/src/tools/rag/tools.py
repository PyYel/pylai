from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from pylcloud.database.src.search.DatabaseSearch import DatabaseSearch
from pylcloud.gpt.src.GPT import GPT

from ..tool import Tool
from pylai.logger import _config_logger

class Document(BaseModel):
    """
    Reduced document schema containing only the critical information
    needed by an LLM, preventing context window bloat.
    """

    id: str = Field(..., description="The unique identifier of the document chunk.")
    content: str = Field(..., description="The text content of the matching chunk.")
    source: str = Field("", description="The origin document or URI of this chunk.")
    score: Optional[float] = Field(
        None, description="The similarity score of the match."
    )


class RAGSearchTool(Tool):
    """
    Semantically searches a pylcloud vector store.

    ``DatabaseSearch.similarity_search`` is the one method whose signature is
    actually consistent across pylcloud's search backends (Elasticsearch,
    Opensearch, S3Vector), so this works with any of them interchangeably.
    """

    NAME = "search_knowledge_base"
    DESCRIPTION = (
        "Searches the vector knowledge base for chunks semantically relevant to a "
        "query, returning a cleaned, lightweight list of matches."
    )

    def __init__(
        self,
        db_client: DatabaseSearch,
        embedder: GPT,
        embedding_model_name: str,
        index_name: str,
        **tool_kwargs: Any,
    ) -> None:
        self.db_client = db_client
        self.embedder = embedder
        self.embedding_model_name = embedding_model_name
        self.index_name = index_name
        super().__init__(**tool_kwargs)

        self.logger = _config_logger(logs_name="RAGSearchTool")


    def __call__(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Searches the vector knowledge base for chunks semantically relevant to a query.

        Args:
            query: The natural language query to search the knowledge base with.
            top_k: The number of top matching chunks to return.
        """
        self.logger.info(
            "Searching knowledge base (query=%s, top_k=%d)", query, top_k
        )
        embedding = self.embedder.return_embedding(model_name=self.embedding_model_name, prompt=query)
        if embedding is None:
            return []

        matches = self.db_client.similarity_search(
            index_name=self.index_name,
            vector_query=embedding["embedding"],
            k=top_k,
        )
        return [
            Document(
                id=str(match.get("id", "")),
                content=match.get("metadata", {}).get("content", ""),
                source=match.get("metadata", {}).get("source", ""),
                score=match.get("score"),
            ).model_dump()
            for match in matches
        ]


class RAGIngestTool(Tool):
    """
    Embeds and stores raw text chunks.

    NOTE: unlike ``similarity_search``, pylcloud's ``DatabaseSearch.send_data`` is
    NOT consistently shaped across backends today: ``DatabaseSearchS3Vector`` takes
    a single ``(index_name, data_id, vector, metadata)`` write, while
    ``DatabaseSearchElasticsearch``/``DatabaseSearchOpensearch`` take a bulk
    ``(index_name, documents: list[dict], _ids=None)`` write. This tool is built
    against the S3Vector shape (the natural fit for "embed and store one chunk").
    Using it against an Elasticsearch/Opensearch client will raise a clear error
    instead of silently doing the wrong thing — see the migration review for the
    suggested fix (aligning pylcloud's ``send_data`` signatures).
    """

    NAME = "ingest_knowledge_base_documents"
    DESCRIPTION = "Embeds and stores raw text chunks into the vector knowledge base for later retrieval."

    def __init__(
        self,
        db_client: DatabaseSearch,
        embedder: GPT,
        embedding_model_name: str,
        index_name: str,
        **tool_kwargs: Any,
    ) -> None:
        self.db_client = db_client
        self.embedder = embedder
        self.embedding_model_name = embedding_model_name
        self.index_name = index_name
        super().__init__(**tool_kwargs)
        self.logger = _config_logger(logs_name="RAGIngestTool")

    def __call__(self, documents: List[str], source: str = "") -> str:
        """Embeds and stores raw text chunks into the vector knowledge base for later retrieval.

        Args:
            documents: Raw text chunks to embed and store in the knowledge base.
            source: The origin document or URI shared by all chunks in this batch.
        """
        self.logger.info(
            "Ingesting %d document chunk(s) into '%s'", len(documents), self.index_name
        )
        for chunk in documents:
            embedding = self.embedder.return_embedding(model_name=self.embedding_model_name, prompt=chunk)
            if embedding is None:
                continue
            data_id = self.db_client._hash_content(chunk, prefixes=[source] if source else [])
            try:
                self.db_client.send_data(
                    index_name=self.index_name,
                    data_id=data_id,
                    vector=embedding["embedding"],
                    metadata={"content": chunk, "source": source},
                )
            except TypeError as exc:
                raise TypeError(
                    f"{type(self.db_client).__name__}.send_data does not accept the "
                    "(index_name, data_id, vector, metadata) signature this tool "
                    "was built for. See the RAG tools migration review."
                ) from exc

        return f"Ingested {len(documents)} document chunk(s)."

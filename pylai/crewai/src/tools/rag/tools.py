from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict

from crewai.tools import BaseTool

from pylcloud.database import DatabaseSearchElasticsearch, DatabaseSearchOpensearch, DatabaseSearchS3Vector

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


class SearchInput(BaseModel):
    """Input argument validation schema for RAGSearchTool."""

    query: str = Field(
        ..., description="The natural language query to search the knowledge base with."
    )
    top_k: int = Field(
        default=5, description="The number of top matching chunks to return."
    )


class IngestInput(BaseModel):
    """Input argument validation schema for RAGIngestTool."""

    documents: List[str] = Field(
        ..., description="Raw text chunks to embed and store in the knowledge base."
    )
    source: str = Field(
        default="",
        description="The origin document or URI shared by all chunks in this batch.",
    )


class RAGSearchTool(BaseTool):
    name: str = "search_knowledge_base"
    description: str = (
        "Searches the vector knowledge base for chunks semantically relevant to a query, "
        "returning a cleaned, lightweight list of matches."
    )
    args_schema: type[BaseModel] = SearchInput
    vector_db_interface: VectorDBInterface

    # Required: allows arbitrary non-pydantic types (such as VectorDBInterface) on the BaseTool class
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _run(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        self.vector_db_interface.logger.info(
            "CrewAI executing RAGSearchTool (query: %s, top_k: %d)", query, top_k
        )
        raw_matches = self.vector_db_interface.search(query=query, top_k=top_k)
        return [Document(**match).model_dump() for match in raw_matches]


class RAGIngestTool(BaseTool):
    name: str = "ingest_knowledge_base_documents"
    description: str = (
        "Embeds and stores raw text chunks into the vector knowledge base for later retrieval."
    )
    args_schema: type[BaseModel] = IngestInput
    vector_db_interface: VectorDBInterface

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _run(self, documents: List[str], source: str = "") -> str:
        self.vector_db_interface.logger.info(
            "CrewAI executing RAGIngestTool (documents: %d)", len(documents)
        )
        self.vector_db_interface.upsert(documents=documents, source=source)
        return f"Ingested {len(documents)} document chunk(s)."

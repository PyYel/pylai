from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict

from crewai.tools import BaseTool

from pylcloud.database import DatabaseGraphLocal


class GraphNode(BaseModel):
    """
    Reduced node schema containing only the critical information
    needed by an LLM, preventing context window bloat.
    """

    id: str = Field(..., description="The unique identifier of the node.")
    labels: List[str] = Field(default_factory=list, description="The node's labels.")
    properties: Dict[str, Any] = Field(
        default_factory=dict,
        description="Truncated and key-value filtered subset of the node properties.",
    )


class GraphRelationship(BaseModel):
    """
    Reduced relationship schema containing only the critical information
    needed by an LLM, preventing context window bloat.
    """

    source_id: str = Field(..., description="The unique identifier of the origin node.")
    target_id: str = Field(
        ..., description="The unique identifier of the destination node."
    )
    type: str = Field("RELATED_TO", description="The relationship type.")
    properties: Dict[str, Any] = Field(
        default_factory=dict,
        description="Truncated and key-value filtered subset of the relationship properties.",
    )


class QueryInput(BaseModel):
    """Input argument validation schema for GraphQueryTool."""

    node_id: Optional[str] = Field(
        None, description="Return this single node along with its incident edges."
    )
    labels: Optional[List[str]] = Field(
        None, description="Filter nodes that carry all of these labels."
    )
    properties: Optional[Dict[str, Any]] = Field(
        None, description="Filter nodes whose properties match these key-value pairs."
    )
    edge_type: Optional[str] = Field(None, description="Filter edges of this relationship type.")
    source: Optional[str] = Field(None, description="Filter edges originating from this node id.")
    target: Optional[str] = Field(None, description="Filter edges pointing to this node id.")
    limit: Optional[int] = Field(None, description="Maximum number of matches to return.")


class NeighborsInput(BaseModel):
    """Input argument validation schema for GraphGetNeighborsTool."""

    node_id: str = Field(..., description="The unique identifier of the node to expand.")


class GraphQueryTool(BaseTool):
    name: str = "query_graph_database"
    description: str = (
        "Runs a pattern-matching query against the knowledge graph (by node id, labels, "
        "properties, or edge triple) and returns a cleaned, lightweight set of matching "
        "nodes and edges."
    )
    args_schema: type[BaseModel] = QueryInput
    graph_db_interface: DatabaseGraphLocal

    # Required: allows arbitrary non-pydantic types (such as DatabaseGraphLocal) on the BaseTool class
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _run(
        self,
        node_id: Optional[str] = None,
        labels: Optional[List[str]] = None,
        properties: Optional[Dict[str, Any]] = None,
        edge_type: Optional[str] = None,
        source: Optional[str] = None,
        target: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        self.graph_db_interface.logger.info(
            "CrewAI executing GraphQueryTool (node_id: %s)", node_id
        )
        result = self.graph_db_interface.query_data(
            node_id=node_id,
            labels=labels,
            properties=properties,
            edge_type=edge_type,
            source=source,
            target=target,
            limit=limit,
        )
        nodes = [
            GraphNode(id=nid, labels=n.get("labels", []), properties=n.get("properties", {})).model_dump()
            for nid, n in result["nodes"].items()
        ]
        edges = [
            GraphRelationship(
                source_id=e["source"],
                target_id=e["target"],
                type=e.get("type") or "RELATED_TO",
                properties=e.get("properties", {}),
            ).model_dump()
            for e in result["edges"]
        ]
        return {"nodes": nodes, "edges": edges}


class GraphGetNeighborsTool(BaseTool):
    name: str = "get_graph_neighbors"
    description: str = (
        "Retrieves the edges directly incident to a given node, restricted to a lightweight schema."
    )
    args_schema: type[BaseModel] = NeighborsInput
    graph_db_interface: DatabaseGraphLocal

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _run(self, node_id: str) -> List[Dict[str, Any]]:
        self.graph_db_interface.logger.info(
            "CrewAI executing GraphGetNeighborsTool (node_id: %s)", node_id
        )
        result = self.graph_db_interface.query_data(node_id=node_id)
        return [
            GraphRelationship(
                source_id=e["source"],
                target_id=e["target"],
                type=e.get("type") or "RELATED_TO",
                properties=e.get("properties", {}),
            ).model_dump()
            for e in result["edges"]
        ]

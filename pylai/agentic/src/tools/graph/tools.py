from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from pydantic_ai import Tool

from pylcloud.database.src.graph.DatabaseGraph import DatabaseGraph


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


def build_graph_query_tool(db_client: DatabaseGraph) -> Tool:
    """Builds a pydantic-ai Tool that pattern-matches a pylcloud knowledge graph."""

    def query_graph_database(
        node_id: Optional[str] = None,
        labels: Optional[List[str]] = None,
        properties: Optional[Dict[str, Any]] = None,
        edge_type: Optional[str] = None,
        source: Optional[str] = None,
        target: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Runs a pattern-matching query against the knowledge graph.

        Args:
            node_id: Return this single node along with its incident edges.
            labels: Filter nodes that carry all of these labels.
            properties: Filter nodes whose properties match these key-value pairs.
            edge_type: Filter edges of this relationship type.
            source: Filter edges originating from this node id.
            target: Filter edges pointing to this node id.
            limit: Maximum number of matches to return.
        """
        db_client.logger.info("Querying graph database (node_id=%s)", node_id)
        result = db_client.query_data(
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

    return Tool(
        query_graph_database,
        name="query_graph_database",
        description=(
            "Runs a pattern-matching query against the knowledge graph (by node id, labels, "
            "properties, or edge triple) and returns a cleaned, lightweight set of matching "
            "nodes and edges."
        ),
    )


def build_graph_neighbors_tool(db_client: DatabaseGraph) -> Tool:
    """Builds a pydantic-ai Tool that expands a node's direct neighborhood."""

    def get_graph_neighbors(node_id: str) -> List[Dict[str, Any]]:
        """Retrieves the edges directly incident to a given node.

        Args:
            node_id: The unique identifier of the node to expand.
        """
        db_client.logger.info("Fetching graph neighbors (node_id=%s)", node_id)
        result = db_client.query_data(node_id=node_id)
        return [
            GraphRelationship(
                source_id=e["source"],
                target_id=e["target"],
                type=e.get("type") or "RELATED_TO",
                properties=e.get("properties", {}),
            ).model_dump()
            for e in result["edges"]
        ]

    return Tool(
        get_graph_neighbors,
        name="get_graph_neighbors",
        description="Retrieves the edges directly incident to a given node, restricted to a lightweight schema.",
    )

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .src.agents.agent import Agent
    from .src.agents.orchestrator.agent import OrchestratorAgent
    from .src.agents.rag.agent import RAGAgent
    from .src.agents.graph.agent import GraphAgent
    from .src.tools.tool import Tool
    from .src.tools.rag.tools import RAGIngestTool, RAGSearchTool
    from .src.tools.graph.tools import GraphQueryTool, GraphNeighborsTool
    from .src.utils import test_mcp_server_sync, cli_local_chatbot, simple_local_chatbot

_LAZY_IMPORTS = {
    "Agent": ".src.agents.agent",
    "OrchestratorAgent": ".src.agents.orchestrator.agent",
    "RAGAgent": ".src.agents.rag.agent",
    "GraphAgent": ".src.agents.graph.agent",
    "Tool": ".src.tools.tool",
    "RAGIngestTool": ".src.tools.rag.tools",
    "RAGSearchTool": ".src.tools.rag.tools",
    "GraphQueryTool": ".src.tools.graph.tools",
    "GraphNeighborsTool": ".src.tools.graph.tools",
    "test_mcp_server_sync": ".src.utils",
    "cli_local_chatbot": ".src.utils",
    "simple_local_chatbot": ".src.utils",
}

_EXTRAS = {
    "RAGAgent": "rag",
    "RAGIngestTool": "rag",
    "RAGSearchTool": "rag",
    "GraphAgent": "graph",
    "GraphQueryTool": "graph",
    "GraphNeighborsTool": "graph",
}

__all__ = list(_LAZY_IMPORTS)


def __getattr__(name):
    module_path = _LAZY_IMPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        module = import_module(module_path, __name__)
    except ImportError as e:
        extra = _EXTRAS.get(name)
        if extra is None:
            raise
        raise ImportError(
            f"{name} requires the '{extra}' extra: pip install pylai[{extra}]"
        ) from e
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))

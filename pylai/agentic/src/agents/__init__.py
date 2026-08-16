from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import Agent
    from .rag.agent import RAGAgent
    from .graph.agent import GraphAgent
    from .orchestrator.agent import OrchestratorAgent

_LAZY_IMPORTS = {
    "Agent": ".agent",
    "RAGAgent": ".rag.agent",
    "GraphAgent": ".graph.agent",
    "OrchestratorAgent": ".orchestrator.agent",
}

_EXTRAS = {
    "RAGAgent": "rag",
    "GraphAgent": "graph",
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

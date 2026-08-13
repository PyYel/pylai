from abc import ABC, abstractmethod
from typing import Any, Optional

from pydantic_ai import Tool as PydanticTool

from logger import _config_logger


class Tool(PydanticTool, ABC):
    """
    pylai's base tool: subclasses implement ``__call__`` instead of passing a
    free function to ``Tool(...)``. ``__call__`` can be sync or async and take
    whatever typed parameters the tool needs (plus a leading ``ctx: RunContext``
    when it needs run context, e.g. to delegate to another agent) — pydantic-ai
    builds the tool's JSON schema from ``__call__``'s signature and docstring
    exactly like it would for a plain function; bound methods drop ``self`` from
    that signature automatically, so nothing extra is needed to hide it.

    Using ``__call__`` (rather than e.g. ``_run``) means a tool instance stays a
    plain callable — ``RAGSearchTool(...)("some query")`` works directly,
    without going through an agent — the same shape a bare pydantic-ai tool
    function has, just packaged as a class so it can carry its own dependencies
    (a db client, an embedder, ...) as instance attributes. Inheriting ``ABC``
    means forgetting to implement ``__call__`` on a subclass fails at
    instantiation time, not at first tool-call time.

    ``NAME``/``DESCRIPTION`` class attributes give each subclass its identity;
    both are still overridable per instance via the constructor.
    """

    NAME: str = ""
    DESCRIPTION: str = ""

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        **tool_kwargs: Any,
    ) -> None:
        super().__init__(
            self.__call__,
            name=name or self.NAME or None,
            description=description or self.DESCRIPTION or None,
            **tool_kwargs,
        )

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

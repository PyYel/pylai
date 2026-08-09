import asyncio
from typing import Any, List, Mapping, Optional

from pydantic_ai import Agent, Tool
from pydantic_ai.capabilities import Capability


def agent_as_tool(agent: Agent[Any, Any], *, name: str, description: Optional[str] = None) -> Tool:
    """
    Wraps a pydantic-ai ``Agent`` into a ``Tool`` that runs it — pylai's delegation
    primitive. Any agent can delegate to any other agent by adding a tool built
    this way; there is no separate "team"/"orchestrator" plumbing involved.
    """
    tool_name = f"consult_{name}"
    tool_description = description or agent.description or f"Delegates to the '{name}' agent."

    async def _delegate(query: str) -> Any:
        result = await agent.run(query)
        return result.output

    _delegate.__name__ = tool_name
    _delegate.__doc__ = tool_description

    return Tool(_delegate, name=tool_name, description=tool_description)


def build_broadcast_tool(delegates: Mapping[str, Agent[Any, Any]]) -> Tool:
    """
    Builds ``consult_multiple_agents``: runs a query against several delegate
    agents concurrently via ``asyncio.gather`` and returns once all of them have
    answered. This is the deterministic version of "call several delegates in
    parallel" — it doesn't depend on the model choosing to emit multiple tool
    calls in the same turn on its own (pydantic-ai already runs same-turn tool
    calls concurrently by default, this just makes fan-out reachable in one call).
    """
    delegate_names = list(delegates.keys())

    async def consult_multiple_agents(query: str, agent_names: Optional[List[str]] = None) -> dict:
        """Runs the same query against several delegate agents concurrently and waits for all of them.

        Use this instead of calling delegates one at a time when the query needs
        input from more than one specialist (e.g. a knowledge-base agent, a graph
        agent, and a web agent all contributing to the same answer).

        Args:
            query: The question or task to send to each selected delegate.
            agent_names: Which delegates to consult, by name. Omit to consult all of them.
        """
        targets = agent_names or delegate_names
        unknown = set(targets) - set(delegates)
        if unknown:
            raise ValueError(f"Unknown delegate(s): {sorted(unknown)}. Available: {delegate_names}")

        results = await asyncio.gather(
            *(delegates[target_name].run(query) for target_name in targets),
            return_exceptions=True,
        )
        return {
            target_name: (f"ERROR: {result}" if isinstance(result, BaseException) else result.output)
            for target_name, result in zip(targets, results)
        }

    return Tool(
        consult_multiple_agents,
        name="consult_multiple_agents",
        description=(
            "Runs the same query against several delegate agents in parallel and waits "
            f"for all of them to return. Available delegates: {', '.join(delegate_names)}."
        ),
    )


def delegation_capability(delegates: Mapping[str, Agent[Any, Any]], *, id: str = "delegation") -> Capability:
    """
    Bundles per-delegate tools (and the concurrent broadcast tool, when there are
    2+ delegates) into a single reusable ``pydantic_ai.capabilities.Capability`` —
    the "orchestrator can map agents into tools" behavior, packaged the idiomatic
    pydantic-ai way instead of as a custom base class. Since it's just a
    ``Capability``, it can be attached to any ``Agent`` (``capabilities=[...]``),
    not only ``OrchestratorAgent``.
    """
    tools = [agent_as_tool(agent, name=name) for name, agent in delegates.items()]
    if len(delegates) > 1:
        tools.append(build_broadcast_tool(delegates))

    return Capability(
        id=id,
        tools=tools,
        description=f"Delegates to specialist agents: {', '.join(delegates)}.",
    )

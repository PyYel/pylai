import asyncio
from typing import Any, Dict, List, Optional, Sequence

from pydantic_ai import Agent as PydanticAgent
from pydantic_ai import RunContext
from pydantic_ai import Tool as PydanticTool
from pydantic_ai.capabilities import Capability
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.models import KnownModelName, Model
from abc import ABC

from ..tools import Tool
from pylai.logger import _config_logger


class Agent(PydanticAgent, ABC):
    """
    PyYelAI Base Agent
    """

    DEFAULT_INSTRUCTIONS: str = (
        "You are a helpful agent. Answer to the best of your knowledge, and ask "
        "for more details or information when unsure."
    )
    DEFAULT_DESCRIPTION: str = "A base agent, with tools, designed to answer queries."

    def __init__(
        self,
        model: Model | KnownModelName | str,
        name: str,
        *,
        tools: Sequence[PydanticTool] = (),
        delegates: Sequence[PydanticAgent] = (),
        mcp_urls: Sequence[str] = (),
        mcp_tokens: Sequence[str] = (),
        instructions: Optional[str] = None,
        description: Optional[str] = None,
        **agent_kwargs: Any,
    ) -> None:
        """
        Instanciates an agent. Agents are defined by a set of instructions and tools.

        Parameters
        ----------
        model: str
            The LLM model tied to this specific agent instance execution. An agent running
            long context or complex tasks might need a larger model (resp. smaller), hence this setting.
        tools: Sequence[Tool]
            A list of tools.
        delegates: Sequence[Agent]
            A list of agents this agent is allowed to delegate to (ask things, call just like tools). 
            When empty, then Agent will not be able to call any other agent.
        mcp_urls: Sequence[str]
            A list of fully qualified mcp endpoints (``https://example.domain.com/path/mcp``)
        mcp_tokens: Sequence[str]
            If provided, a list of tokens to use for server side auth. If the MCP hosts list features both secured and 
            unsecured endpoints, make sure to add placeholders strings (``""``), so ``len(mcp_urls)==len(mcp_tokens)``.
        instructions: str
            Instructions to use for this agent. When None, will default to a generic template.
        description: str
            A human-readable description of the agent, attached to the agent. When None, will default to a generic template.
        **agent_kwargs: dict
            Extra argument to pass the the agent constructor.
        """

        self.logger = _config_logger(logs_name="Agent")

        self.model_name = model if isinstance(model, str) else model.model_name
        self.agent_name = name

        self.delegates: Dict[str, "PydanticAgent[Any, Any]"] = self._name_delegates(delegates)
        capabilities: List[Any] = []
        if self.delegates:
            delegate_tools = [
                self._build_delegate_tool(delegate_name, delegate)
                for delegate_name, delegate in self.delegates.items()
            ]
            if len(self.delegates) > 1:
                delegate_tools.append(self._build_multidelegate_tool(self.delegates))
            capabilities.append(Capability(
                id="delegation",
                tools=delegate_tools,
                description=f"Delegates to specialist agents: {', '.join(self.delegates)}.",
            ))

        self.logger.debug(f"Creating instance of agent {self.agent_name} running on model {self.model_name}")
        super().__init__(
            model=model,
            name=name,
            instructions=instructions or self.DEFAULT_INSTRUCTIONS,
            description=description or self.DEFAULT_DESCRIPTION,
            tools=tools,
            toolsets=self._build_mcp_toolsets(mcp_urls, mcp_tokens),
            capabilities=capabilities,
            **agent_kwargs,
        )

    def _build_mcp_toolsets(self, mcp_urls: Sequence[str], mcp_tokens: Sequence[str]) -> List[MCPToolset]:
        """Builds one MCPToolset per (url, token) pair, bearer-authenticated when a token is given."""
        return [
            MCPToolset(
                mcp_url,
                headers={"Authorization": f"Bearer {mcp_token}"} if mcp_token else None,
            )
            for mcp_url, mcp_token in zip(mcp_urls, mcp_tokens)
        ]

    def _name_delegates(
        self, delegates: Sequence["PydanticAgent[Any, Any]"]
    ) -> Dict[str, PydanticAgent[Any, Any]]:
        """
        Maps agent instances to their string names.
        """
        by_name: Dict[str, "PydanticAgent[Any, Any]"] = {}
        for delegate in delegates:
            if not delegate.name:
                raise ValueError(
                    f"Delegate agent {delegate!r} has no `name` set — pylai delegation "
                    "tools are named after their delegate, so every delegate needs one."
                )
            if delegate.name in by_name:
                raise ValueError(
                    f"Two delegates are both named {delegate.name!r} — pylai's ready-made "
                    "agents (RAGAgent, GraphAgent, OrchestratorAgent, ...) default to a "
                    "fixed name, so pass an explicit `name=` when using more than one of "
                    "the same kind as sibling delegates."
                )
            by_name[delegate.name] = delegate
        return by_name

    def _build_delegate_tool(self, name: str, delegate: "PydanticAgent[Any, Any]") -> PydanticTool:
        """
        Builds ``consult_<name>``: runs ``delegate`` and returns its output —
        pydantic-ai's own documented delegation pattern ("Coordinate Multiple
        Agents"), forwarding ``ctx.usage`` so usage/limits accumulate correctly
        across the whole delegation tree rather than resetting per agent.
        """

        async def consult(ctx: RunContext, query: str) -> Any:
            """Delegates a query to this agent and returns its answer.

            Args:
                query: The question to send to the delegate agent.
            """
            self.logger.info(f"Agent {self.agent_name} is consulting agent {name}")
            result = await delegate.run(query, usage=ctx.usage)
            return result.output

        return PydanticTool(
            consult,
            name=f"consult_{name}",
            description=delegate.description or f"Delegates to the '{name}' agent.",
        )

    def _build_multidelegate_tool(self, delegates: Dict[str, "PydanticAgent[Any, Any]"]) -> PydanticTool:
        """
        A tool's tool: allows multi agent delegation in paralel through a single tool call.
        If this method did not exist, upon calling multiple agents at once:
        - Orchestrator needs to call agent A, B, and C
        - Orchestrator would first call call_agent_A tool
        - Agent A would run and call one or many tools
        - Only upon completion, Orchestrator would start calling agent B... Even if non dependant.
        """
        delegate_names = list(delegates)

        async def consult_multiple_agents(
            ctx: RunContext, query: str, agent_names: Optional[List[str]] = None
        ) -> dict:
            """Runs the same query against several delegate agents concurrently and waits for all of them.

            Use this instead of calling delegates one at a time when the query
            needs input from more than one specialist.

            Args:
                query: The question to send to each selected delegate.
                agent_names: Which delegates to consult, by name. Omit to consult all of them.
            """
            targets = agent_names or delegate_names
            unknown = set(targets) - set(delegates)
            if unknown:
                raise ValueError(f"Unknown delegate(s): {sorted(unknown)}. Available: {delegate_names}")

            self.logger.info(f"Agent {self.agent_name} is consulting agents {agent_names}")
            results = await asyncio.gather(
                *(delegates[target_name].run(query, usage=ctx.usage) for target_name in targets),
                return_exceptions=True,
            )
            return {
                target_name: (f"ERROR: {result}" if isinstance(result, BaseException) else result.output)
                for target_name, result in zip(targets, results)
            }

        return PydanticTool(
            consult_multiple_agents,
            name="consult_multiple_agents",
            description=(
                "Runs the same query against several delegate agents in parallel and waits "
                f"for all of them to return. Available delegates: {', '.join(delegate_names)}."
            ),
        )

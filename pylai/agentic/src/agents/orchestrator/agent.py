from typing import Any, List, Mapping

from pydantic_ai import Agent
from pydantic_ai.models import KnownModelName, Model

from .._instructions import role_instructions
from ...tools.delegation.tools import delegation_capability


class OrchestratorAgent(Agent):
    """
    Conversational front-door agent: answers directly when it can, and delegates
    to specialist agents when the query needs their tools.

    Delegation is added as a ``pydantic_ai.capabilities.Capability`` (see
    ``tools/delegation/tools.py``) bundling one tool per delegate plus a
    concurrent ``consult_multiple_agents`` broadcast tool when there are 2+ — the
    idiomatic pydantic-ai way to package "reusable tools + instructions" rather
    than a custom base class.

    Because ``delegates`` accepts any ``pydantic_ai.Agent`` — including another
    ``OrchestratorAgent`` — hierarchies fall out for free, with no separate
    "team" type and no depth limit beyond what the model can usefully reason
    about:

        leaf_rag = RAGAgent(...)
        leaf_graph = GraphAgent(...)
        research_orchestrator = OrchestratorAgent(
            model=model, delegates={"rag_agent": leaf_rag, "graph_agent": leaf_graph}
        )
        root_orchestrator = OrchestratorAgent(
            model=model, delegates={"research_orchestrator": research_orchestrator}
        )
    """

    DEFAULT_ROLE = "Assistant Chatbot"
    DEFAULT_GOAL = "Interact conversationally with the user. Answer general questions directly."
    DEFAULT_BACKSTORY = (
        "You coordinate specialized agents and present synthesized responses to the user. "
        "If this is casual conversation or a general question, answer directly. "
        "If the query requires specific data, delegate to the relevant specialist tool — "
        "consult several of them concurrently with consult_multiple_agents when the answer "
        "needs input from more than one specialist. "
        "Feel free to ask the user for more details before delegating or using tools, "
        "when the request is ambiguous."
    )

    def __init__(
        self,
        *,
        model: Model | KnownModelName | str,
        delegates: Mapping[str, Agent] = {},
        role: str = DEFAULT_ROLE,
        goal: str = DEFAULT_GOAL,
        backstory: str = DEFAULT_BACKSTORY,
        **agent_kwargs: Any,
    ) -> None:
        self.delegates = dict(delegates)

        capabilities: List[Any] = list(agent_kwargs.pop("capabilities", None) or [])
        if self.delegates:
            capabilities.append(delegation_capability(self.delegates))

        super().__init__(
            model=model,
            name="orchestrator_agent",
            description=goal,
            instructions=role_instructions(role, goal, backstory),
            capabilities=capabilities or None,
            **agent_kwargs,
        )

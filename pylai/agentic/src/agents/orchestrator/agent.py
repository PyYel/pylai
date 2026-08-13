from typing import Any, Optional, Sequence, Union

from pydantic_ai import Agent as PydanticAgent
from pydantic_ai.models import KnownModelName, Model

from ..agent import Agent


class OrchestratorAgent(Agent):
    """
    PylAI Agent
    """

    DEFAULT_INSTRUCTIONS = (
        "Answer directly when the question is general or conversational. When it "
        "needs specialized data, delegate to the relevant agent tool instead of "
        "guessing. When it needs input from more than one specialist, use "
        "consult_multiple_agents to gather them concurrently rather than calling "
        "them one at a time. Ask the user for clarification before delegating an "
        "ambiguous request."
    )
    DEFAULT_DESCRIPTION = "Coordinates specialist agents and answers directly otherwise."

    def __init__(
        self,
        model: Union[Model, KnownModelName],
        name: str = "OrchestratorAgent",
        delegates: Union[PydanticAgent, Sequence[PydanticAgent]] = (),
        instructions: Optional[str] = None,
        description: Optional[str] = None,
        **agent_kwargs: Any,
    ) -> None:
        """
        Instanciates an ``OrchestratorAgent``.

        Most notable, the ready to use orchestrator does not have tools: it 
        either calls agents for help (if ``delegates`` is not empty), or answers
        directly to the user when no agents seem relevant.

        As always, you may tweak this behaviour by overiding the agent
        instructions.

        Parameters
        ----------
        model: str
            The LLM model tied to this agent execution.
        """
        super().__init__(
            model=model,
            name=name,
            delegates=delegates if isinstance(delegates, Sequence) else [delegates],
            instructions=instructions,
            description=description,
            **agent_kwargs,
        )

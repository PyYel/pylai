import os
import sys

from dotenv import load_dotenv
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

MODULE_DIR_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if __name__ == "__main__":
    sys.path.append(MODULE_DIR_PATH)

load_dotenv(os.path.join(MODULE_DIR_PATH, ".env"), override=False)

from .agent import OrchestratorAgent  # noqa: E402

INFERENCE_MODEL_NAME = os.getenv("INFERENCE_MODEL_NAME", "")
INFERENCE_ENDPOINT = os.getenv("INFERENCE_ENDPOINT", "")
INFERENCE_API_KEY = os.getenv("INFERENCE_API_KEY", "")


def start_chatbot():
    """Interactive smoke test for a bare OrchestratorAgent (no delegates wired).

    See ``test/main.py`` for the orchestrator wired up with RAG/graph delegates,
    including a two-level hierarchy.
    """
    model = OpenAIChatModel(
        INFERENCE_MODEL_NAME,
        provider=OpenAIProvider(base_url=INFERENCE_ENDPOINT, api_key=INFERENCE_API_KEY),
    )

    agent = OrchestratorAgent(model=model)

    print("=== Interactive pydantic-ai Orchestrator Chatbot Started ===")
    agent.to_cli_sync(prog_name="orchestrator_agent")


if __name__ == "__main__":
    start_chatbot()

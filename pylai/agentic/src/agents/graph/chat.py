import os
import sys

from dotenv import load_dotenv
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from pylcloud.database import DatabaseGraphLocal

MODULE_DIR_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if __name__ == "__main__":
    sys.path.append(MODULE_DIR_PATH)

load_dotenv(os.path.join(MODULE_DIR_PATH, ".env"), override=False)

from .agent import GraphAgent  # noqa: E402

INFERENCE_MODEL_NAME = os.getenv("INFERENCE_MODEL_NAME", "")
INFERENCE_ENDPOINT = os.getenv("INFERENCE_ENDPOINT", "")
INFERENCE_API_KEY = os.getenv("INFERENCE_API_KEY", "")
DB_HOST = os.getenv("DB_HOST", "")


def start_chatbot():
    """Interactive smoke test for GraphAgent, wired to the local (in-memory/file) graph backend."""
    model = OpenAIChatModel(
        INFERENCE_MODEL_NAME,
        provider=OpenAIProvider(base_url=INFERENCE_ENDPOINT, api_key=INFERENCE_API_KEY),
    )

    db_client = DatabaseGraphLocal(host=DB_HOST)
    db_client.connect_database(host=DB_HOST)

    agent = GraphAgent(db_client=db_client, model=model)

    print("=== Interactive pydantic-ai Graph Chatbot Started ===")
    agent.to_cli_sync(prog_name="graph_agent")


if __name__ == "__main__":
    start_chatbot()

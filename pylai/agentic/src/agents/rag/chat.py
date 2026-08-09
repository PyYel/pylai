import os
import sys

from dotenv import load_dotenv
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from pylcloud.database import DatabaseSearchElasticsearch
from pylcloud.gpt import GPTAzure

MODULE_DIR_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if __name__ == "__main__":
    sys.path.append(MODULE_DIR_PATH)

load_dotenv(os.path.join(MODULE_DIR_PATH, ".env"), override=False)

from .agent import RAGAgent  # noqa: E402

INFERENCE_MODEL_NAME = os.getenv("INFERENCE_MODEL_NAME", "")
INFERENCE_ENDPOINT = os.getenv("INFERENCE_ENDPOINT", "")
INFERENCE_API_KEY = os.getenv("INFERENCE_API_KEY", "")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "")
RAG_INDEX_NAME = os.getenv("RAG_INDEX_NAME", "")

DB_HOST = os.getenv("DB_HOST", "")
DB_USER = os.getenv("DB_USER", "")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")


def start_chatbot():
    """Interactive smoke test for RAGAgent, wired to an Elasticsearch backend.

    Swap ``DatabaseSearchElasticsearch`` for ``DatabaseSearchOpensearch`` or
    ``DatabaseSearchS3Vector`` to test against a different search backend.
    """
    model = OpenAIChatModel(
        INFERENCE_MODEL_NAME,
        provider=OpenAIProvider(base_url=INFERENCE_ENDPOINT, api_key=INFERENCE_API_KEY),
    )

    db_client = DatabaseSearchElasticsearch(host=DB_HOST, user=DB_USER, password=DB_PASSWORD)
    embedder = GPTAzure()

    agent = RAGAgent(
        db_client=db_client,
        embedder=embedder,
        index_name=RAG_INDEX_NAME,
        embedding_model_name=EMBEDDING_MODEL_NAME,
        model=model,
    )

    print("=== Interactive pydantic-ai RAG Chatbot Started ===")
    agent.to_cli_sync(prog_name="rag_agent")


if __name__ == "__main__":
    start_chatbot()

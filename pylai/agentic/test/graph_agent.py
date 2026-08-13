import os, sys

MAIN_DIR = os.path.dirname(os.path.dirname((os.path.dirname(__file__))))
if __name__ == "__main__":
    sys.path.append(MAIN_DIR)

from agentic import cli_local_chatbot, simple_local_chatbot
from agentic import OrchestratorAgent, GraphAgent
from pylcloud.database import DatabaseGraphLocal

os.environ["LOGS_LEVEL"] = "DEBUG"

db_client = DatabaseGraphLocal(
    host="tree.ggraph.json"
)
db_client.connect_database(host="tree.ggraph.json")
graph_agent = GraphAgent(
    db_client=db_client,
    model="bedrock:eu.amazon.nova-micro-v1:0",
)

simple_local_chatbot(agent=graph_agent)

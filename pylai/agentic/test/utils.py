import os, sys

MAIN_DIR = os.path.dirname(os.path.dirname((os.path.dirname(__file__))))
if __name__ == "__main__":
    sys.path.append(MAIN_DIR)

from agentic import start_local_chatbot
from agentic import OrchestratorAgent

agent = OrchestratorAgent(
    model="bedrock:eu.amazon.nova-micro-v1:0"
)
start_local_chatbot(agent=agent)




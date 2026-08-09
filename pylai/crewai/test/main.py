import os, sys

from pylai.crewai import OrchestratorAgent, RAGAgent, GraphAgent

# This is a test file. Run it with pyyel as an installed package.

for cls in [
    RAGAgent,
    GraphAgent,
    OrchestratorAgent,
]:
    try:
        cls()
    except Exception as e:
        print(e)

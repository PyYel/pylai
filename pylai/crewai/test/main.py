import os, sys

from pylai.crewai import OrchestratorAgent, RAGAgent

# This is a test file. Run it with pyyel as an installed package.

for cls in [
    RAGAgent,
    OrchestratorAgent,
]:
    try:
        cls()
    except Exception as e:
        print(e)

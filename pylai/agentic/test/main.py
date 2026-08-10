import os, sys

from pylai.agentic import *

# This is a test file. Run it with pyyel as an installed package.

for cls in [
    Agent,
    GraphAgent,
    OrchestratorAgent,
    RAGSearchTool,
    RAGIngestTool,
    GraphQueryTool,
    GraphNeighborsTool,
]:
    try:
        cls()
    except Exception as e:
        if "required positional argument" not in str(e):
            print(e)

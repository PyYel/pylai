"""
Smoke + functional test for pylai.agentic.

1. Constructs every public agent against real (non-networked) pylcloud fixtures,
   to catch import/wiring errors early.
2. Drives a two-level orchestrator hierarchy with pydantic-ai's ``TestModel``
   (which calls every tool the model can see) to prove delegation actually
   works end to end, not just that construction doesn't raise.

Run with pylai installed as a package.
"""

import os
import tempfile

from pydantic_ai import Agent
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.providers.openai import OpenAIProvider

from pylai.agentic import GraphAgent, OrchestratorAgent, RAGAgent
from pylcloud.database import DatabaseGraphLocal, DatabaseSearchElasticsearch
from pylcloud.gpt import GPTAzure

model = OpenAIChatModel(
    "gpt-4o-mini",
    provider=OpenAIProvider(base_url="http://localhost:0", api_key="dummy"),
)

graph_db_client = DatabaseGraphLocal(host=os.path.join(tempfile.gettempdir(), "pylai_smoke_test.ggraph.json"))
graph_db_client.connect_database(host=graph_db_client.host)

rag_db_client = DatabaseSearchElasticsearch(host="https://localhost:9200")
embedder = GPTAzure()

fixtures = {}
fixtures["RAGAgent"] = lambda: RAGAgent(
    db_client=rag_db_client,
    embedder=embedder,
    index_name="pylai-smoke-test",
    embedding_model_name="text-embedding-3-small",
    model=model,
)
fixtures["GraphAgent"] = lambda: GraphAgent(db_client=graph_db_client, model=model)
fixtures["OrchestratorAgent (no delegates)"] = lambda: OrchestratorAgent(model=model)

for label, build in fixtures.items():
    try:
        build()
        print(f"OK: {label}")
    except Exception as e:
        print(f"FAILED: {label}: {e}")


def test_delegation_end_to_end():
    """
    Builds orchestrator -> sub_orchestrator -> {agent_a, agent_b} entirely on
    TestModel (which calls every tool it can see) and asserts the delegate
    tools were actually invoked, and that consult_multiple_agents only exists
    where there are 2+ delegates.
    """
    agent_a = Agent(TestModel(), name="agent_a", description="Leaf agent A", instructions="You are agent A.")
    agent_b = Agent(TestModel(), name="agent_b", description="Leaf agent B", instructions="You are agent B.")

    sub_orchestrator = OrchestratorAgent(
        model=TestModel(), delegates={"agent_a": agent_a, "agent_b": agent_b}
    )
    root_orchestrator = OrchestratorAgent(
        model=TestModel(), delegates={"research_team": sub_orchestrator}
    )

    result = root_orchestrator.run_sync("Gather info from both specialists and summarize.")
    tool_calls = {
        part.tool_name
        for msg in result.all_messages()
        for part in msg.parts
        if isinstance(part, ToolCallPart)
    }

    assert "consult_research_team" in tool_calls, "root_orchestrator did not delegate to sub_orchestrator"
    assert "consult_multiple_agents" not in tool_calls, "root_orchestrator has only 1 delegate, should have no broadcast tool"
    print(f"OK: delegation end-to-end (root_orchestrator called: {sorted(tool_calls)})")

    # sub_orchestrator has 2 delegates: run it directly to confirm its own tools.
    sub_result = sub_orchestrator.run_sync("Gather info from both specialists.")
    sub_tool_calls = {
        part.tool_name
        for msg in sub_result.all_messages()
        for part in msg.parts
        if isinstance(part, ToolCallPart)
    }
    assert {"consult_agent_a", "consult_agent_b", "consult_multiple_agents"} <= sub_tool_calls
    print(f"OK: sub_orchestrator broadcast wiring (called: {sorted(sub_tool_calls)})")


try:
    test_delegation_end_to_end()
except Exception as e:
    print(f"FAILED: delegation end-to-end: {e}")

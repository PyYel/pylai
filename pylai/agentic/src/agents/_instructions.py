from typing import List, Sequence

from pydantic_ai.mcp import MCPToolset


def role_instructions(role: str, goal: str, backstory: str = "") -> str:
    """
    Folds pylai's role/goal/backstory persona concept (kept from the CrewAI-era
    API for familiarity) into plain agent ``instructions`` — pydantic-ai itself
    has no persona notion.
    """
    parts = [f"Role: {role}", f"Goal: {goal}"]
    if backstory:
        parts.append(f"Backstory: {backstory}")
    return "\n".join(parts)


def build_mcp_toolsets(mcp_urls: Sequence[str], mcp_tokens: Sequence[str]) -> List[MCPToolset]:
    """Builds one MCPToolset per (url, token) pair, bearer-authenticated when a token is given."""
    return [
        MCPToolset(
            mcp_url,
            headers={"Authorization": f"Bearer {mcp_token}"} if mcp_token else None,
        )
        for mcp_url, mcp_token in zip(mcp_urls, mcp_tokens)
    ]

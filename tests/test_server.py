"""End-to-end tests through an in-process MCP client."""

import json

import pytest
from mcp import Client
from mcp_types import CreateMessageResult, TextContent

from prompt_improver import server


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def no_api_credentials(monkeypatch):
    monkeypatch.setattr(server, "anthropic_credentials_available", lambda: False)
    monkeypatch.delenv("PROMPT_IMPROVER_BACKEND", raising=False)


HISTORY = [
    {"role": "user", "content": "Write a Python function that parses ISO dates."},
    {"role": "assistant", "content": "def parse(s): return datetime.strptime(s, '%Y-%m-%d')"},
]


@pytest.mark.anyio
async def test_host_mode_without_sampling():
    async with Client(server.mcp) as client:
        result = await client.call_tool("improve_prompt", {"prompt": "that doesn't work with timezones, fix it",
                                                           "history": HISTORY})
    assert not result.is_error
    data = result.structured_content
    assert data["backend_used"] == "host"
    assert "failure_feedback" in [t["id"] for t in data["techniques"]]
    assert "parses ISO dates" in data["improved_prompt"]


@pytest.mark.anyio
async def test_sampling_backend_when_client_supports_it():
    seen = {}

    async def sampling_callback(context, params):
        seen["system"] = params.system_prompt
        seen["user"] = params.messages[0].content.text
        reply = {"improved_prompt": "Rewritten by the client model.", "techniques_applied": ["failure_feedback"],
                 "changes": ["Added the timezone requirement."], "assumptions": [], "clarifying_questions": []}
        return CreateMessageResult(role="assistant", content=TextContent(type="text", text=json.dumps(reply)),
                                   model="fake-model", stop_reason="endTurn")

    async with Client(server.mcp, sampling_callback=sampling_callback) as client:
        result = await client.call_tool("improve_prompt", {"prompt": "that doesn't work with timezones, fix it",
                                                           "history": HISTORY})
    assert not result.is_error, result.content
    data = result.structured_content
    assert data["backend_used"] == "sampling"
    assert data["improved_prompt"] == "Rewritten by the client model."
    assert "Technique catalog" in seen["system"]
    assert "parses ISO dates" in seen["user"]


@pytest.mark.anyio
async def test_unknown_technique_is_a_tool_error():
    async with Client(server.mcp) as client:
        result = await client.call_tool("improve_prompt", {"prompt": "hi", "techniques": ["telepathy"]})
    assert result.is_error
    assert "Unknown technique" in result.content[0].text


@pytest.mark.anyio
async def test_analyze_and_list_and_slash_prompt():
    async with Client(server.mcp) as client:
        analysis = await client.call_tool("analyze_prompt", {"prompt": "What does this chart show?", "has_image": True})
        catalog = await client.call_tool("list_techniques", {})
        prompt = await client.get_prompt("improve", {"request": "explain why the sky is blue"})
    assert "visual_grounding" in [t["id"] for t in analysis.structured_content["techniques"]]
    assert any("2411.10440" in p for t in catalog.structured_content["result"] for p in t["papers"])
    text = prompt.messages[0].content.text
    assert "explain why the sky is blue" in text and "Step-back" in text

"""Launch the installed server over stdio, as an MCP client would, and check it answers."""

import asyncio
import sys

from mcp import Client, StdioServerParameters


async def main() -> None:
    params = StdioServerParameters(command=sys.executable, args=["-m", "prompt_improver"])
    async with Client(params) as client:
        tools = sorted(t.name for t in (await client.list_tools()).tools)
        assert tools == ["analyze_prompt", "improve_prompt", "list_techniques"], tools
        result = await client.call_tool("improve_prompt", {"prompt": "What is in this photo?", "backend": "host"})
        assert not result.is_error and result.structured_content["backend_used"] == "host"
    print("stdio smoke test passed:", ", ".join(tools))


asyncio.run(main())

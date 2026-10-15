"""MCP server exposing the prompt improver (stdio transport)."""

from __future__ import annotations

import os
from importlib.metadata import PackageNotFoundError, version
from typing import Annotated, Any, Literal

from mcp.server.mcpserver import Context, MCPServer, Resolve, Sample
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import CreateMessageResult, ModelPreferences, SamplingMessage, TextContent
from pydantic import Field

from prompt_improver import compose, engine
from prompt_improver.analyzer import profile_task, select_techniques
from prompt_improver.llm import AnthropicRewriter, anthropic_credentials_available
from prompt_improver.models import Analysis, Backend, ImprovedPrompt, TechniqueInfo, Turn
from prompt_improver.techniques import PROMPT_TECHNIQUE_IDS, TECHNIQUES, Target

INSTRUCTIONS = """Rewrites prompts using techniques from the prompting literature (LLaVA-CoT staged reasoning, System 2 \
Attention, Rephrase-and-Respond, Chain-of-Verification, Step-Back, Plan-and-Solve, Least-to-Most, and others).
Use improve_prompt when the user asks to improve, rewrite, or sharpen a prompt, or asks you to improve their request \
before answering it. The server cannot see the conversation: pass the user's prompt verbatim, the earlier turns it \
depends on (including your previous responses), and any other context. When backend_used is "host", the result is a \
template draft: finish it by following host_instructions before showing or running it."""

try:
    __version__ = version("prompt-improver-mcp")
except PackageNotFoundError:
    __version__ = "0.0.0"

mcp = MCPServer("prompt-improver", instructions=INSTRUCTIONS, version=__version__)

TECHNIQUE_IDS_DOC = ", ".join(PROMPT_TECHNIQUE_IDS)


def _turns(history: Any) -> list[Turn]:
    return [t if isinstance(t, Turn) else Turn.model_validate(t) for t in history or []]


def _client_can_sample(ctx: Context | None) -> bool:
    try:
        caps = ctx.client_capabilities if ctx is not None else None
    except ValueError:
        return False
    return caps is not None and caps.sampling is not None


def choose_backend(requested: str, ctx: Context | None) -> str:
    choice = requested if requested != "auto" else os.environ.get("PROMPT_IMPROVER_BACKEND", "auto")
    if choice != "auto":
        return choice
    if anthropic_credentials_available():
        return "anthropic"
    if _client_can_sample(ctx):
        return "sampling"
    return "host"


def _request(
    prompt: str,
    context: str,
    history: Any,
    target: Target,
    has_image: bool,
    techniques: list[str] | None,
    refine_rounds: int = 0,
) -> engine.ImproveRequest:
    return engine.ImproveRequest(
        prompt=prompt,
        context=context,
        history=_turns(history),
        target=target,
        has_image=has_image,
        techniques=techniques,
        refine_rounds=refine_rounds,
    )


async def _sample_rewrite(
    ctx: Context,
    prompt: str,
    context: str = "",
    history: Any = None,
    target: Target = "reasoning",
    has_image: bool = False,
    techniques: list[str] | None = None,
    backend: Backend = "auto",
) -> Sample | None:
    """Ask the client's LLM for the rewrite when sampling is the chosen backend; otherwise resolve to None."""
    if choose_backend(backend, ctx) != "sampling" or not _client_can_sample(ctx):
        return None
    req = _request(prompt, context, history, target, has_image, techniques)
    try:
        system, user = engine.rewriter_messages(req, engine.plan(req))
    except ValueError:
        return None  # invalid technique ids; the tool body reports the error
    return Sample(
        [SamplingMessage(role="user", content=TextContent(type="text", text=f"{user}\n\n{compose.JSON_REPLY_INSTRUCTION}"))],
        max_tokens=8000,
        system_prompt=system,
        model_preferences=ModelPreferences(intelligence_priority=0.8, speed_priority=0.3),
    )


def _sampled_text(result: CreateMessageResult) -> str:
    blocks = result.content if isinstance(result.content, list) else [result.content]
    return "".join(b.text for b in blocks if isinstance(b, TextContent))


@mcp.tool(
    title="Improve a prompt",
    description=(
        "Rewrite a prompt so a model answers it better, using techniques from the prompting literature: LLaVA-CoT "
        "staged reasoning, System 2 Attention context distillation, conversational query rewriting, "
        "Rephrase-and-Respond, Chain-of-Verification, Step-Back, Plan-and-Solve, Least-to-Most, and more. The "
        "server cannot see the conversation, so pass the current prompt verbatim plus the earlier turns and context "
        "it depends on. Returns the improved prompt, the techniques used (with paper citations), what changed, "
        "assumptions, and clarifying questions. If backend_used is 'host', improved_prompt is a template draft: "
        "finish it by following host_instructions."
    ),
)
async def improve_prompt(
    ctx: Context,
    sampled: Annotated[CreateMessageResult | None, Resolve(_sample_rewrite)],
    prompt: Annotated[str, Field(description="The user's current prompt, verbatim.")],
    context: Annotated[
        str, Field(description="Other material the prompt depends on: files, documents, task background.")
    ] = "",
    history: Annotated[
        list[Turn] | None,
        Field(description="Earlier conversation turns, oldest first, including previous assistant responses."),
    ] = None,
    target: Annotated[
        Target,
        Field(
            description="'reasoning' for models with built-in thinking (Claude, o-series, R1): no visible "
            "chain-of-thought cues. 'standard' for models without it (e.g. small open VLMs): adds explicit "
            "step-by-step and LLaVA-CoT stage tags."
        ),
    ] = "reasoning",
    has_image: Annotated[bool, Field(description="True if the prompt will be sent with an image or video.")] = False,
    techniques: Annotated[
        list[str] | None,
        Field(description=f"Force exactly these techniques instead of automatic selection. Ids: {TECHNIQUE_IDS_DOC}"),
    ] = None,
    refine_rounds: Annotated[
        int, Field(ge=0, le=2, description="Self-Refine critique-and-revise rounds (Claude API backend only).")
    ] = 0,
    backend: Annotated[
        Backend,
        Field(
            description="'auto' uses the Claude API if credentials are available, else MCP sampling if the client "
            "supports it, else 'host' (a template draft for the calling model to finish)."
        ),
    ] = "auto",
) -> ImprovedPrompt:
    try:
        req = _request(prompt, context, history, target, has_image, techniques, refine_rounds)
        engine.plan(req)
    except ValueError as e:
        raise ToolError(str(e)) from e

    choice = choose_backend(backend, ctx)
    if choice == "anthropic":
        return await engine.improve_with_anthropic(req, AnthropicRewriter())
    if choice == "sampling":
        if sampled is None:
            return engine.improve_on_host(
                req, "Sampling was requested but this client does not support it; returned a host-mode draft."
            )
        return engine.improve_from_sample(req, _sampled_text(sampled), sampled.model)
    return engine.improve_on_host(req)


@mcp.tool(
    title="Analyze a prompt",
    description="Profile a prompt's task type and list the techniques improve_prompt would apply, with citations, "
    "without rewriting anything.",
)
def analyze_prompt(
    prompt: Annotated[str, Field(description="The prompt to analyze, verbatim.")],
    context: str = "",
    history: list[Turn] | None = None,
    target: Target = "reasoning",
    has_image: bool = False,
) -> Analysis:
    return engine.plan(_request(prompt, context, history, target, has_image, None)).analysis()


@mcp.tool(title="List techniques", description="List every technique the improver knows, with the papers behind it.")
def list_techniques() -> list[TechniqueInfo]:
    return [
        TechniqueInfo(
            id=t.id,
            name=t.name,
            kind=t.kind,
            when=t.when,
            how=t.how,
            how_for_reasoning_models=t.how_reasoning,
            papers=[p.cite() for p in t.papers],
        )
        for t in TECHNIQUES
    ]


@mcp.prompt(
    name="improve",
    title="Improve a prompt",
    description="Improve a prompt using this conversation as context, then offer to run it. Pass the prompt as the "
    "argument, or leave it empty to improve your most recent message.",
)
def improve_command(request: str = "") -> str:
    target: Literal["reasoning"] = "reasoning"
    profile = profile_task(request, "", [], False)
    picks = [
        p
        for p in select_techniques(profile, target)
        if p.technique.id not in ("context_distillation", "self_contained_rewrite")
    ]
    return compose.slash_command_text(request, picks, target)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()

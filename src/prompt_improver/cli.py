"""Command-line interface: improve a prompt without an MCP client."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from prompt_improver import engine
from prompt_improver.llm import AnthropicRewriter, anthropic_credentials_available
from prompt_improver.models import ImprovedPrompt, Turn
from prompt_improver.techniques import PROMPT_TECHNIQUE_IDS


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="prompt-improver",
        description="Rewrite a prompt using techniques from the prompting literature (LLaVA-CoT, System 2 "
        "Attention, Chain-of-Verification, ...).",
    )
    p.add_argument("prompt", help="The prompt to improve, or - to read it from stdin.")
    p.add_argument("--context", metavar="FILE", help="File with context the prompt depends on.")
    p.add_argument(
        "--history",
        metavar="FILE",
        help='JSON file with earlier turns, oldest first: [{"role": "user" | "assistant", "content": "..."}].',
    )
    p.add_argument(
        "--target",
        choices=["reasoning", "standard"],
        default="reasoning",
        help="reasoning: models with built-in thinking (default). standard: models without it; adds explicit "
        "step-by-step cues and LLaVA-CoT stage tags.",
    )
    p.add_argument("--image", action="store_true", help="The prompt will be sent with an image or video.")
    p.add_argument(
        "--technique",
        dest="techniques",
        action="append",
        choices=PROMPT_TECHNIQUE_IDS,
        metavar="ID",
        help=f"Force a technique instead of automatic selection (repeatable). One of: {', '.join(PROMPT_TECHNIQUE_IDS)}.",
    )
    p.add_argument("--refine", type=int, choices=[0, 1, 2], default=0, help="Self-Refine rounds (Claude API only).")
    p.add_argument(
        "--backend",
        choices=["auto", "anthropic", "host"],
        default="auto",
        help="auto uses the Claude API when credentials are available, otherwise a template draft (host).",
    )
    p.add_argument("--json", action="store_true", help="Print the full result as JSON.")
    return p.parse_args(argv)


def _request(args: argparse.Namespace) -> engine.ImproveRequest:
    prompt = sys.stdin.read() if args.prompt == "-" else args.prompt
    context = Path(args.context).read_text(encoding="utf-8") if args.context else ""
    history: list[Turn] = []
    if args.history:
        raw = json.loads(Path(args.history).read_text(encoding="utf-8"))
        history = [Turn.model_validate(t) for t in raw]
    return engine.ImproveRequest(
        prompt=prompt,
        context=context,
        history=history,
        target=args.target,
        has_image=args.image,
        techniques=args.techniques,
        refine_rounds=args.refine,
    )


async def _run(args: argparse.Namespace, req: engine.ImproveRequest) -> ImprovedPrompt:
    if args.backend == "anthropic" or (args.backend == "auto" and anthropic_credentials_available()):
        return await engine.improve_with_anthropic(req, AnthropicRewriter())
    return engine.improve_on_host(req)


def render(result: ImprovedPrompt) -> str:
    lines = [result.improved_prompt, "", f"-- backend: {result.backend_used} | target: {result.target}", "-- techniques:"]
    lines += [f"   * {t.name} [{t.citation}]: {t.why}" for t in result.techniques]
    for label, items in (
        ("assumptions", result.assumptions),
        ("clarifying questions", result.clarifying_questions),
        ("inference tips", result.inference_tips),
        ("notes", result.notes),
    ):
        if items:
            lines.append(f"-- {label}:")
            lines += [f"   * {item}" for item in items]
    if result.host_instructions:
        lines += [
            "-- this is a template draft. To finish it, give it to any LLM with the instructions below, or set "
            "ANTHROPIC_API_KEY so the Claude API does the rewrite:",
            result.host_instructions,
        ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        req = _request(args)
        engine.plan(req)
    except (OSError, json.JSONDecodeError, ValidationError, ValueError) as e:
        print(f"prompt-improver: error: {e}", file=sys.stderr)
        return 2
    result = asyncio.run(_run(args, req))
    print(result.model_dump_json(indent=2) if args.json else render(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())

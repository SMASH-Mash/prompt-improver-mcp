"""The improvement pipeline: profile the task, pick techniques, rewrite, optionally self-refine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from prompt_improver import compose
from prompt_improver.analyzer import Pick, inference_tips, profile_task, select_techniques
from prompt_improver.distill import Distilled, distill
from prompt_improver.llm import AnthropicRewriter, RewriteError, parse_json_reply
from prompt_improver.models import Analysis, ImprovedPrompt, TaskProfile, TechniqueRef, Turn
from prompt_improver.techniques import BY_ID, Target


@dataclass
class ImproveRequest:
    prompt: str
    context: str = ""
    history: list[Turn] = field(default_factory=list)
    target: Target = "reasoning"
    has_image: bool = False
    techniques: list[str] | None = None
    refine_rounds: int = 0


@dataclass
class Plan:
    profile: TaskProfile
    picks: list[Pick]
    tips: list[str]

    def analysis(self) -> Analysis:
        return Analysis(profile=self.profile, techniques=[p.ref() for p in self.picks], inference_tips=self.tips)


def plan(req: ImproveRequest) -> Plan:
    profile = profile_task(req.prompt, req.context, req.history, req.has_image)
    picks = select_techniques(profile, req.target, req.techniques)
    return Plan(profile, picks, inference_tips(profile, picks, req.target))


def rewriter_messages(req: ImproveRequest, p: Plan) -> tuple[str, str]:
    """(system, user) for one rewrite call; deterministic for a given request."""
    user = compose.rewrite_request(req.prompt, req.context, req.history, p.profile, p.picks, req.target)
    return compose.REWRITER_SYSTEM, user


def _technique_refs(ids: list[str], p: Plan) -> tuple[list[TechniqueRef], list[str]]:
    why = {pick.technique.id: pick.why for pick in p.picks}
    refs, unknown = [], []
    for tid in dict.fromkeys(ids):
        t = BY_ID.get(tid)
        if t is None:
            unknown.append(tid)
            continue
        refs.append(TechniqueRef(id=t.id, name=t.name, citation=t.citation(), why=why.get(tid, "chosen by the rewriter")))
    return refs, unknown


def _from_llm(
    data: dict[str, Any], backend: Literal["anthropic", "sampling"], req: ImproveRequest, p: Plan, notes: list[str]
) -> ImprovedPrompt:
    refs, unknown = _technique_refs(list(data.get("techniques_applied") or []), p)
    if unknown:
        notes.append(f"Rewriter named unknown technique ids, ignored: {', '.join(unknown)}")
    if data.get("critique"):
        notes.append(f"Self-refine critique: {data['critique']}")
    return ImprovedPrompt(
        improved_prompt=data["improved_prompt"],
        backend_used=backend,
        target=req.target,
        profile=p.profile,
        techniques=refs,
        changes=list(data.get("changes") or []),
        assumptions=list(data.get("assumptions") or []),
        clarifying_questions=list(data.get("clarifying_questions") or []),
        inference_tips=p.tips,
        notes=notes,
    )


def host_result(req: ImproveRequest, p: Plan, notes: list[str]) -> ImprovedPrompt:
    distilled: Distilled = distill(
        req.prompt,
        req.context,
        req.history,
        keep_last_exchange=p.profile.references_history or p.profile.dissatisfied_with_previous,
    )
    draft = compose.build_draft(req.prompt, req.history, p.profile, p.picks, distilled, req.target)
    return ImprovedPrompt(
        improved_prompt=draft,
        backend_used="host",
        target=req.target,
        profile=p.profile,
        techniques=[pick.ref() for pick in p.picks],
        changes=[f"{pick.technique.name}: {pick.why}" for pick in p.picks],
        assumptions=[],
        clarifying_questions=[],
        inference_tips=p.tips,
        host_instructions=compose.host_instructions(p.profile, p.picks, distilled, req.target),
        notes=notes,
    )


async def improve_with_anthropic(req: ImproveRequest, rewriter: AnthropicRewriter) -> ImprovedPrompt:
    p = plan(req)
    system, user = rewriter_messages(req, p)
    notes: list[str] = []
    try:
        data = await rewriter.run(system, user, compose.REWRITE_SCHEMA)
        for _ in range(req.refine_rounds):
            data = await rewriter.run(system, compose.refine_request(user, data), compose.REFINE_SCHEMA)
    except RewriteError as e:
        notes.append(f"Claude API rewrite failed, fell back to host mode: {e}")
        return host_result(req, p, notes)
    notes.append(f"Rewritten by {rewriter.model} (effort {rewriter.effort}).")
    return _from_llm(data, "anthropic", req, p, notes)


def improve_from_sample(req: ImproveRequest, sampled_text: str, model: str | None) -> ImprovedPrompt:
    p = plan(req)
    notes = [f"Rewritten via MCP sampling by the client's model{f' ({model})' if model else ''}."]
    if req.refine_rounds:
        notes.append("refine_rounds is not supported with sampling; skipped.")
    data = parse_json_reply(sampled_text)
    if data is None:
        notes.append("The sampled reply was not the expected JSON, fell back to host mode.")
        return host_result(req, p, notes)
    return _from_llm(data, "sampling", req, p, notes)


def improve_on_host(req: ImproveRequest, reason: str | None = None) -> ImprovedPrompt:
    return host_result(req, plan(req), [reason] if reason else [])

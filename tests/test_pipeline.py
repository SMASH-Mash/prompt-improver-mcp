import json
from types import SimpleNamespace

import pytest

from prompt_improver import compose, engine
from prompt_improver.analyzer import profile_task, select_techniques
from prompt_improver.distill import distill
from prompt_improver.llm import AnthropicRewriter, parse_json_reply
from prompt_improver.models import Turn
from prompt_improver.techniques import BY_ID, TECHNIQUES


def ids(picks):
    return [p.technique.id for p in picks]


def test_every_technique_cites_an_arxiv_paper():
    for t in TECHNIQUES:
        assert t.papers and all(p.arxiv for p in t.papers), t.id
    assert BY_ID["staged_reasoning"].papers[0].arxiv == "2411.10440"  # LLaVA-CoT


def test_visual_question_gets_llava_cot_techniques():
    profile = profile_task("How many objects are left in the image after removing the purple ones?", "", [])
    assert "visual" in profile.task_types
    picked = ids(select_techniques(profile, "reasoning"))
    assert "visual_grounding" in picked and "staged_reasoning" in picked
    assert picked[-1] == "specificity_format"


def test_figure_out_is_not_visual():
    assert "visual" not in profile_task("Help me figure out why the cache misses", "", []).task_types


def test_zero_shot_cot_only_for_standard_targets():
    profile = profile_task("Solve: a train travels 120 km in 1.5 hours; what is its speed?", "", [])
    assert "zero_shot_cot" not in ids(select_techniques(profile, "reasoning"))
    assert "zero_shot_cot" in ids(select_techniques(profile, "standard"))


def test_correction_after_previous_answer():
    history = [
        Turn(role="user", content="Write a regex that matches ISO dates."),
        Turn(role="assistant", content=r"Use \d{4}-\d{2}-\d{2}"),
    ]
    profile = profile_task("No, that still matches 2024-13-45, fix it", "", history)
    assert profile.dissatisfied_with_previous and profile.references_history
    picked = ids(select_techniques(profile, "reasoning"))
    assert picked[:3] == ["context_distillation", "self_contained_rewrite", "failure_feedback"]


def test_short_prompt_without_context_is_ambiguous():
    profile = profile_task("make it better", "", [])
    assert profile.ambiguity == "high"
    assert {"rephrase_expand", "clarify"} <= set(ids(select_techniques(profile, "reasoning")))


def test_forced_techniques_are_validated():
    profile = profile_task("anything", "", [])
    assert ids(select_techniques(profile, "reasoning", ["step_back"])) == ["step_back"]
    with pytest.raises(ValueError, match="Unknown technique"):
        select_techniques(profile, "reasoning", ["telepathy"])


def test_distill_keeps_relevant_segments_and_last_exchange():
    context = "The billing service retries failed charges.\n\nOur office plants need watering on Fridays."
    history = [
        Turn(role="user", content="Why do charges fail twice?"),
        Turn(role="assistant", content="Because the retry loop has no idempotency key."),
    ]
    out = distill("add an idempotency key to the billing retry", context, history, keep_last_exchange=True)
    texts = [s.text for s in out.segments]
    assert any("billing service" in t for t in texts)
    assert not any("plants" in t for t in texts)
    assert any("idempotency key" in t for t in texts)
    assert out.omitted == 1


def test_draft_respects_target():
    profile = profile_task("What is the area of the triangle in this diagram?", "", [])
    for target, has_tags in (("standard", True), ("reasoning", False)):
        picks = select_techniques(profile, target)
        draft = compose.build_draft("What is the area of the triangle in this diagram?", [], profile, picks,
                                    distill("", "", []), target)
        assert ("<REASONING>" in draft) is has_tags
        assert "<task>" in draft and "<output_format>" in draft


def test_host_mode_rewrites_correction_into_self_contained_task():
    history = [
        Turn(role="user", content="Summarize the attached paper in 3 bullets."),
        Turn(role="assistant", content="- It studies X\n- It finds Y\n- It concludes Z"),
    ]
    result = engine.improve_on_host(engine.ImproveRequest(prompt="that's not what I asked, use 5 bullets", history=history))
    assert result.backend_used == "host"
    assert result.improved_prompt.count("Summarize the attached paper") == 1
    assert "It concludes Z" in result.improved_prompt
    assert result.host_instructions and "template draft" in result.host_instructions


def test_follow_up_takes_task_type_from_referenced_turn():
    history = [
        Turn(role="user", content="Refactor handler.py so the retry loop is testable."),
        Turn(role="assistant", content="Here is a version with dependency injection."),
    ]
    assert "code" in profile_task("that broke it, try again", "", history).task_types


def test_host_instructions_preview_omitted_context():
    req = engine.ImproveRequest(prompt="add an idempotency key to the billing retry",
                                context="Billing retry runs hourly.\n\nWe deploy on AWS Lambda.")
    result = engine.improve_on_host(req)
    assert "We deploy on AWS Lambda" in result.host_instructions


def test_parse_json_reply_handles_fences_and_garbage():
    assert parse_json_reply('Sure!\n```json\n{"improved_prompt": "x"}\n```')["improved_prompt"] == "x"
    assert parse_json_reply("no json here") is None
    assert parse_json_reply('{"other": 1}') is None


class FakeMessages:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def fake_client(*replies):
    messages = FakeMessages(replies)
    return SimpleNamespace(beta=SimpleNamespace(messages=messages)), messages


def reply(data=None, stop_reason="end_turn"):
    text = json.dumps(data) if data is not None else ""
    return SimpleNamespace(
        stop_reason=stop_reason,
        stop_details=SimpleNamespace(category="cyber") if stop_reason == "refusal" else None,
        content=[SimpleNamespace(type="text", text=text)],
    )


REWRITE = {
    "improved_prompt": "Improved.",
    "techniques_applied": ["visual_grounding", "made_up"],
    "changes": ["Grounded the answer in the image."],
    "assumptions": [],
    "clarifying_questions": [],
}


@pytest.mark.anyio
async def test_anthropic_backend_request_shape_and_refine(monkeypatch):
    monkeypatch.delenv("PROMPT_IMPROVER_MODEL", raising=False)
    client, messages = fake_client(reply(REWRITE), reply({**REWRITE, "critique": "Fine.", "improved_prompt": "Refined."}))
    req = engine.ImproveRequest(prompt="What breed is the dog in this photo?", refine_rounds=1)
    result = await engine.improve_with_anthropic(req, AnthropicRewriter(client=client))

    first = messages.calls[0]
    assert first["model"] == "claude-opus-5-5"
    assert first["fallbacks"] == "default" and first["betas"] == ["server-side-fallback-2026-07-01"]
    assert first["output_config"]["format"]["schema"] == compose.REWRITE_SCHEMA
    assert "<current_prompt>" in first["messages"][0]["content"]
    assert messages.calls[1]["output_config"]["format"]["schema"] == compose.REFINE_SCHEMA

    assert result.backend_used == "anthropic"
    assert result.improved_prompt == "Refined."
    assert [t.id for t in result.techniques] == ["visual_grounding"]
    assert any("made_up" in n for n in result.notes) and any("critique" in n for n in result.notes)


@pytest.mark.anyio
async def test_anthropic_refusal_falls_back_to_host():
    client, _ = fake_client(reply(stop_reason="refusal"))
    result = await engine.improve_with_anthropic(engine.ImproveRequest(prompt="explain quicksort"), AnthropicRewriter(client=client))
    assert result.backend_used == "host"
    assert any("declined" in n for n in result.notes)


@pytest.fixture
def anyio_backend():
    return "asyncio"

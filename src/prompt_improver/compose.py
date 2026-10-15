"""Builds the texts the improver produces: the rewriter's meta-prompt and the offline template draft."""

from __future__ import annotations

import copy
import json

from prompt_improver.analyzer import Pick
from prompt_improver.distill import LABELS, Distilled
from prompt_improver.models import TaskProfile, Turn
from prompt_improver.techniques import BPO, LLAVA_COT, PE2, Target, catalog_text

REWRITE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "improved_prompt": {"type": "string"},
        "techniques_applied": {"type": "array", "items": {"type": "string"}},
        "changes": {"type": "array", "items": {"type": "string"}},
        "assumptions": {"type": "array", "items": {"type": "string"}},
        "clarifying_questions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["improved_prompt", "techniques_applied", "changes", "assumptions", "clarifying_questions"],
    "additionalProperties": False,
}

REFINE_SCHEMA: dict = copy.deepcopy(REWRITE_SCHEMA)
REFINE_SCHEMA["properties"] = {"critique": {"type": "string"}, **REFINE_SCHEMA["properties"]}
REFINE_SCHEMA["required"] = ["critique", *REWRITE_SCHEMA["required"]]

# Stable across calls so it can be prompt-cached. Rules follow BPO (rewrite the prompt to serve the
# user's intent, never to change it) and PE2 (give the rewriter the task context, a step-by-step
# procedure, and a technique catalog to reason over).
REWRITER_SYSTEM = f"""You are a prompt engineer. You rewrite a user's prompt so that a target language model understands \
and answers it better, while preserving exactly what the user wants.

Rules:
1. Fidelity first. Keep the user's goal, hard constraints, quoted text, and language. Never add requirements, facts, or \
preferences the user did not state or clearly imply. Where you must fill a gap, write it into the prompt as an explicit \
default the model may override, and list it under assumptions.
2. Self-contained. The target model will see only your improved prompt, not the conversation. Resolve references to \
earlier turns and include the context the task depends on, verbatim where precision matters (code, numbers, names).
3. Proportionate. A one-line question should not become a page. Add only the structure that helps this particular task.
4. Target-aware. If the target is "reasoning" (a model with built-in thinking), do not ask it to print step-by-step \
reasoning or to show its thinking. Structure the deliverable instead. Explicit step-by-step cues are only for "standard" \
targets.
5. Ready to send. Write improved_prompt in the user's voice, addressed to the model, with no commentary about the \
rewrite. Use XML tags to separate supplied context from instructions when there is context.
6. Clarifying questions only when different answers would materially change the result: at most three. Otherwise return \
an empty list.

Procedure: work out what the user actually wants (using the history to resolve references and to see what a previous \
answer got wrong), decide which techniques fit, then write the prompt. The heuristic analysis you receive is a \
suggestion: drop techniques that don't fit and add catalog techniques that clearly do.

Technique catalog (use the ids in techniques_applied):
{catalog_text()}

Methodology sources: {BPO.cite()}; {PE2.cite()}."""

JSON_REPLY_INSTRUCTION = (
    "Reply with only a JSON object with these keys: improved_prompt (string), techniques_applied (array of "
    "technique ids), changes (array of strings: what changed and why), assumptions (array of strings), "
    "clarifying_questions (array of strings)."
)


def _xml_escape_tag_text(text: str) -> str:
    return text.replace("</", "<\\/")


def _history_block(history: list[Turn]) -> str:
    turns = "\n".join(f'<turn role="{t.role}">\n{_xml_escape_tag_text(t.content)}\n</turn>' for t in history)
    return f"<conversation_history>\n{turns}\n</conversation_history>"


def rewrite_request(
    prompt: str, context: str, history: list[Turn], profile: TaskProfile, picks: list[Pick], target: Target
) -> str:
    """The user message for the rewriter: inputs plus the heuristic analysis."""
    suggestions = "\n".join(f"- {p.technique.id}: {p.why}" for p in picks)
    parts = [
        f"<target_model>{target}</target_model>",
        "<heuristic_analysis>\n"
        f"task types: {', '.join(profile.task_types)}\n"
        f"signals: {'; '.join(profile.signals) or 'none'}\n"
        f"ambiguity: {profile.ambiguity}\n"
        f"suggested techniques:\n{suggestions}\n"
        "</heuristic_analysis>",
    ]
    if context.strip():
        parts.append(f"<context>\n{_xml_escape_tag_text(context)}\n</context>")
    if history:
        parts.append(_history_block(history))
    parts.append(f"<current_prompt>\n{_xml_escape_tag_text(prompt)}\n</current_prompt>")
    parts.append("Rewrite <current_prompt> following the rules.")
    return "\n\n".join(parts)


def refine_request(base_request: str, candidate: dict) -> str:
    """Self-Refine round: critique the candidate rewrite against the rules, then revise it."""
    return (
        f"{base_request}\n\nA first rewrite produced this candidate:\n"
        f"<candidate>\n{json.dumps(candidate, ensure_ascii=False, indent=2)}\n</candidate>\n\n"
        "Critique the candidate against the rules: fidelity to the user's intent (anything added the user did not "
        "ask for?), self-containedness, clarity, fit of the techniques to this task and target, and proportionate "
        "length. Put the critique in the critique field, then return a revised version in the other fields. If the "
        "candidate is already good, return it unchanged."
    )


def _evidence_source(profile: TaskProfile) -> str:
    return "the image" if "visual" in profile.task_types else "the problem statement and context"


_APPROACH = {
    "visual_grounding": "First note what in the image matters for the question: the relevant objects, their "
    "attributes and how they relate to each other, any visible text or numbers, and the region to focus on. If "
    "something you need is not visible or legible, say so instead of guessing.",
    "step_back": "Start by identifying the general principle or concept that governs this question, then apply it "
    "to the specifics.",
    "least_to_most": "Break the problem into sub-questions ordered so that each builds on the previous answers, and "
    "solve them in that order.",
    "verification": "After drafting, check each key factual claim (names, numbers, dates, citations) on its own, "
    "and correct or drop any that do not hold up. Mark anything you are unsure of rather than stating it as fact.",
}
_PLAN_AND_SOLVE = {
    "standard": "Extract the relevant quantities and devise a plan, then carry out the plan step by step, checking "
    "each calculation.",
    "reasoning": "Extract the relevant quantities and plan before solving. Double-check the calculations, and "
    "report the key intermediate results with the final answer.",
}


def _output_format(profile: TaskProfile, chosen: set[str], target: Target) -> str:
    source = _evidence_source(profile)
    if "staged_reasoning" in chosen and target == "standard":
        return (
            f"Answer in four stages, each wrapped in its tag (the {LLAVA_COT.authors} {LLAVA_COT.year} format):\n"
            "<SUMMARY>Restate the problem and outline how you will solve it.</SUMMARY>\n"
            f"<CAPTION>Describe the facts from {source} that the answer depends on.</CAPTION>\n"
            "<REASONING>Work through the solution step by step.</REASONING>\n"
            "<CONCLUSION>Give the final answer only.</CONCLUSION>"
        )
    if "staged_reasoning" in chosen:
        return (
            "Structure your answer as:\n"
            "1. Approach: one or two sentences on how you will tackle it.\n"
            f"2. Evidence: the specific facts from {source} that the answer relies on.\n"
            "3. Answer: the conclusion, stated directly."
        )
    if "zero_shot_cot" in chosen:
        return "Think step by step, then give the final answer on its own line, starting with \"Answer:\"."
    return "Start with the direct answer or deliverable, then any explanation that is needed."


def build_draft(
    prompt: str,
    history: list[Turn],
    profile: TaskProfile,
    picks: list[Pick],
    distilled: Distilled,
    target: Target,
) -> str:
    """Template-assembled improved prompt used when no LLM is available to rewrite it."""
    chosen = {p.technique.id for p in picks}
    sections: list[str] = []

    last_user = next((i for i in range(len(history) - 1, -1, -1) if history[i].role == "user"), None)
    earlier_request = history[last_user].content if last_user is not None else None
    restate_earlier = "failure_feedback" in chosen and earlier_request is not None
    # When the task restates the earlier request, don't repeat it in the context block.
    shown = [s for s in distilled.segments if not (restate_earlier and s.source == "user" and s.turn == last_user)]
    if shown:
        snippets = "\n\n".join(f"[{LABELS[s.source]}]\n{s.text}" for s in shown)
        sections.append(f"<context>\n{snippets}\n</context>")

    if restate_earlier:
        sections.append(
            "<task>\n"
            f"Earlier request:\n{earlier_request.strip()}\n\n"
            f"My feedback on your previous answer:\n{prompt.strip()}\n\n"
            "Work out specifically what fell short in the previous answer, keep what was correct, and fix the "
            "problem rather than repeating the earlier answer.\n"
            "</task>"
        )
    else:
        sections.append(f"<task>\n{prompt.strip()}\n</task>")

    approach = [_APPROACH[t] for t in ("visual_grounding", "step_back", "least_to_most") if t in chosen]
    if "plan_and_solve" in chosen:
        approach.append(_PLAN_AND_SOLVE[target])
    if "verification" in chosen:
        approach.append(_APPROACH["verification"])
    if "clarify" in chosen or "rephrase_expand" in chosen:
        approach.append("If a detail you need is missing, state the assumption you are making in one line, then "
                        "answer under that assumption.")
    if approach:
        sections.append("<approach>\n" + "\n".join(f"- {a}" for a in approach) + "\n</approach>")

    sections.append(f"<output_format>\n{_output_format(profile, chosen, target)}\n</output_format>")
    if "re_reading" in chosen:
        sections.append(f"To restate the task: {prompt.strip()}")
    return "\n\n".join(sections)


def host_instructions(profile: TaskProfile, picks: list[Pick], distilled: Distilled, target: Target) -> str:
    steps = [
        "No LLM backend rewrote this prompt, so improved_prompt is a template draft. Finish it yourself, using the "
        "conversation you can see, before showing or running it:"
    ]
    if profile.references_history:
        steps.append(
            f"Replace the references to earlier turns ({', '.join(profile.history_references)}) with what they "
            "actually refer to, so the prompt stands alone."
        )
    if distilled.omitted:
        def preview(text: str) -> str:
            flat = " ".join(text.split())
            return flat if len(flat) <= 70 else f"{flat[:70]}..."

        previews = "; ".join(f'[{LABELS[s.source]}] "{preview(s.text)}"' for s in distilled.omitted_segments[:6])
        steps.append(
            "The <context> block was selected by word overlap with the prompt, which misses relevance that isn't "
            f"lexical. Left out: {previews}. Add back any the task depends on, and remove anything irrelevant."
        )
    for p in picks:
        steps.append(f"{p.technique.name} [{p.technique.citation()}]: {p.technique.instruction(target)}")
    steps.append(
        "Keep the user's intent and constraints exactly; do not add requirements. Keep it proportionate to the task."
    )
    if target == "reasoning":
        steps.append("The target model reasons internally, so do not ask it to print step-by-step reasoning.")
    steps.append(
        "Then show the user the final prompt. If a missing detail would materially change the answer, ask up to "
        "three clarifying questions first."
    )
    return "\n".join(f"{i}. {s}" if i else s for i, s in enumerate(steps))


def slash_command_text(request: str, picks: list[Pick], target: Target) -> str:
    """Instructions for the host model when the user runs the `improve` MCP prompt as a slash command."""
    subject = (
        f"<prompt_to_improve>\n{request.strip()}\n</prompt_to_improve>"
        if request.strip()
        else "The prompt to improve is my most recent message before this command."
    )
    techniques = "\n".join(
        f"- {p.technique.name} [{p.technique.citation()}]: {p.technique.instruction(target)}" for p in picks
    )
    reasoning_rule = (
        "\n- The target model reasons internally, so don't ask it to print step-by-step reasoning." if target ==
        "reasoning" else ""
    )
    return f"""Improve a prompt before anything is run. Our conversation so far is its context: it is what words \
like "it", "that" or "the above" refer to, and it shows what a previous answer may have got wrong.

{subject}

Apply the techniques that fit (suggested from the prompt text; skip any that don't apply):
- System 2 Attention context distillation [Weston & Sukhbaatar 2023, arXiv:2311.11829]: include only the parts of our \
conversation the task depends on, verbatim where precision matters.
- Conversational query rewriting [Ma et al. 2023, arXiv:2305.14283]: resolve references to earlier turns so the prompt \
stands alone.
{techniques}

Rules:
- Keep my intent and constraints exactly; never add requirements I didn't state or clearly imply.
- Keep it proportionate: a short question should stay short.{reasoning_rule}

Reply with:
1. The improved prompt in a fenced block.
2. A short list of what you changed and why, naming the technique.
3. Up to three clarifying questions, only if the answers would materially change the result.
Then ask whether to run the improved prompt."""

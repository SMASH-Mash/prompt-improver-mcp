"""Catalog of prompting techniques the improver can apply, each tied to the paper it comes from.

`how` is the rewriting instruction given to whoever rewrites the prompt (the Claude API, the
client's LLM via sampling, or the host model). `how_reasoning` overrides it when the target
model has built-in thinking: such models reason internally, and asking them to print their
reasoning adds nothing and can trigger refusals, so those variants structure the deliverable
instead of requesting a reasoning transcript.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Target = Literal["reasoning", "standard"]


@dataclass(frozen=True)
class Paper:
    authors: str
    year: int
    title: str
    arxiv: str
    venue: str = ""

    def cite(self) -> str:
        venue = f", {self.venue}" if self.venue else ""
        return f"{self.authors} ({self.year}{venue}). {self.title}. arXiv:{self.arxiv}"

    def short(self) -> str:
        return f"{self.authors} {self.year}, arXiv:{self.arxiv}"


@dataclass(frozen=True)
class Technique:
    id: str
    name: str
    kind: Literal["prompt", "inference"]
    papers: tuple[Paper, ...]
    when: str
    how: str
    how_reasoning: str | None = None

    def instruction(self, target: Target) -> str:
        if target == "reasoning" and self.how_reasoning:
            return self.how_reasoning
        return self.how

    def citation(self) -> str:
        return "; ".join(p.short() for p in self.papers)


# All arXiv IDs below were checked against the arXiv API.
LLAVA_COT = Paper("Xu et al.", 2024, "LLaVA-CoT: Let Vision Language Models Reason Step-by-Step", "2411.10440", "ICCV 2025")
S2A = Paper("Weston & Sukhbaatar", 2023, "System 2 Attention (is something you might need too)", "2311.11829")
QUERY_REWRITE = Paper("Ma et al.", 2023, "Query Rewriting for Retrieval-Augmented Large Language Models", "2305.14283", "EMNLP 2023")
RAR = Paper("Deng et al.", 2023, "Rephrase and Respond: Let Large Language Models Ask Better Questions for Themselves", "2311.04205")
CLAM = Paper("Kuhn et al.", 2022, "CLAM: Selective Clarification for Ambiguous Questions with Generative Language Models", "2212.07769")
PROTEGI = Paper("Pryzant et al.", 2023, 'Automatic Prompt Optimization with "Gradient Descent" and Beam Search', "2305.03495", "EMNLP 2023")
SELF_REFINE = Paper("Madaan et al.", 2023, "Self-Refine: Iterative Refinement with Self-Feedback", "2303.17651")
CCOT = Paper("Mitra et al.", 2023, "Compositional Chain-of-Thought Prompting for Large Multimodal Models", "2311.17076")
VISUAL_COT = Paper("Shao et al.", 2024, "Visual CoT: Advancing Multi-Modal Language Models with a Comprehensive Dataset and Benchmark for Chain-of-Thought Reasoning", "2403.16999")
MM_COT = Paper("Zhang et al.", 2023, "Multimodal Chain-of-Thought Reasoning in Language Models", "2302.00923", "TMLR")
STEP_BACK = Paper("Zheng et al.", 2023, "Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models", "2310.06117", "ICLR 2024")
PLAN_SOLVE = Paper("Wang et al.", 2023, "Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models", "2305.04091", "ACL 2023")
LEAST_TO_MOST = Paper("Zhou et al.", 2022, "Least-to-Most Prompting Enables Complex Reasoning in Large Language Models", "2205.10625", "ICLR 2023")
COVE = Paper("Dhuliawala et al.", 2023, "Chain-of-Verification Reduces Hallucination in Large Language Models", "2309.11495")
RE2 = Paper("Xu et al.", 2023, "Re-Reading Improves Reasoning in Large Language Models", "2309.06275", "EMNLP 2024")
PRINCIPLED = Paper("Bsharat et al.", 2023, "Principled Instructions Are All You Need for Questioning LLaMA-1/2, GPT-3.5/4", "2312.16171")
PROMPT_REPORT = Paper("Schulhoff et al.", 2024, "The Prompt Report: A Systematic Survey of Prompt Engineering Techniques", "2406.06608")
COT = Paper("Wei et al.", 2022, "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models", "2201.11903")
ZERO_SHOT_COT = Paper("Kojima et al.", 2022, "Large Language Models are Zero-Shot Reasoners", "2205.11916", "NeurIPS 2022")
SELF_CONSISTENCY = Paper("Wang et al.", 2022, "Self-Consistency Improves Chain of Thought Reasoning in Language Models", "2203.11171", "ICLR 2023")
TOT = Paper("Yao et al.", 2023, "Tree of Thoughts: Deliberate Problem Solving with Large Language Models", "2305.10601", "NeurIPS 2023")
# Framework-level influences on the rewriting procedure itself (cited in the meta-prompt and README).
BPO = Paper("Cheng et al.", 2023, "Black-Box Prompt Optimization: Aligning Large Language Models without Model Training", "2311.04155", "ACL 2024")
PE2 = Paper("Ye et al.", 2023, "Prompt Engineering a Prompt Engineer", "2311.05661", "ACL 2024 Findings")

TECHNIQUES: tuple[Technique, ...] = (
    Technique(
        id="context_distillation",
        name="System 2 Attention context distillation",
        kind="prompt",
        papers=(S2A,),
        when="Context or conversation history is supplied, especially when it is long or contains opinions, "
        "superseded drafts, or leading suggestions.",
        how="Regenerate the context so it keeps only what the current request depends on: facts, constraints, "
        "definitions, and decisions already made. Drop chit-chat, superseded attempts, and opinions or leading "
        "suggestions unless the user stated them as requirements. Quote facts and code verbatim.",
    ),
    Technique(
        id="self_contained_rewrite",
        name="Conversational query rewriting",
        kind="prompt",
        papers=(QUERY_REWRITE, RAR),
        when="The prompt refers to earlier turns ('it', 'that', 'the above', 'same as before', 'again') or is "
        "elliptical, so it cannot be understood on its own.",
        how="Rewrite the prompt so it stands alone: replace pronouns and references to earlier turns with the "
        "concrete things they refer to, and carry forward constraints the user set earlier that still apply.",
    ),
    Technique(
        id="failure_feedback",
        name="Textual-gradient correction from the previous response",
        kind="prompt",
        papers=(PROTEGI, SELF_REFINE),
        when="The user is correcting or retrying after an earlier response ('no', 'wrong', 'still fails', "
        "'instead').",
        how="Diagnose concretely what the previous response got wrong relative to the user's intent (the "
        "'textual gradient'), then add explicit constraints that prevent that failure. Say what to keep from the "
        "previous attempt and what must change; do not ask for the earlier answer to be repeated.",
    ),
    Technique(
        id="rephrase_expand",
        name="Rephrase and Respond",
        kind="prompt",
        papers=(RAR,),
        when="The request is short, vague, or underspecified.",
        how="Rephrase and expand the request: state the goal, the inputs, the scope, and what a good answer looks "
        "like. Expand only what the user stated or clearly implied; never add new requirements.",
    ),
    Technique(
        id="clarify",
        name="Selective clarification",
        kind="prompt",
        papers=(CLAM,),
        when="Key details are missing and different reasonable readings would produce materially different "
        "answers.",
        how="Identify at most three clarifying questions whose answers would change the result most. In the prompt "
        "itself, state the default assumption the model should use for each, so it can proceed without an answer.",
    ),
    Technique(
        id="visual_grounding",
        name="Grounded visual description before reasoning",
        kind="prompt",
        papers=(LLAVA_COT, CCOT, VISUAL_COT, MM_COT),
        when="An image, screenshot, chart, diagram, or video frame is part of the input.",
        how="Before any reasoning, have the model inventory what is visible that matters for the question: the "
        "relevant objects, their attributes and spatial relations (a compact scene graph), any text or numbers in "
        "the image, and the region to focus on. The answer must rest on these observations, and the model must say "
        "when something it needs is not visible or legible instead of guessing.",
    ),
    Technique(
        id="step_back",
        name="Step-back abstraction",
        kind="prompt",
        papers=(STEP_BACK,),
        when="Knowledge-intensive questions (STEM, law, economics, technical design) where a governing principle "
        "or concept determines the answer.",
        how="Have the model first state the general principle, concept, or rule that governs the question, then "
        "apply it to the specifics.",
    ),
    Technique(
        id="plan_and_solve",
        name="Plan-and-Solve",
        kind="prompt",
        papers=(PLAN_SOLVE,),
        when="Math word problems, calculations, and multi-step procedures.",
        how="Ask the model to extract the relevant quantities, devise a plan, then carry out the plan step by step, "
        "checking each calculation.",
        how_reasoning="Ask the model to extract the relevant quantities and plan before solving, to double-check "
        "calculations, and to report the key intermediate results alongside the final answer.",
    ),
    Technique(
        id="least_to_most",
        name="Least-to-Most decomposition",
        kind="prompt",
        papers=(LEAST_TO_MOST,),
        when="Compositional problems or requests with several dependent parts.",
        how="Decompose the problem into an ordered list of sub-questions where each builds on the previous answers, "
        "then solve them in that order.",
    ),
    Technique(
        id="staged_reasoning",
        name="LLaVA-CoT staged reasoning",
        kind="prompt",
        papers=(LLAVA_COT,),
        when="Multi-step reasoning, especially visual question answering, where the answer benefits from an "
        "explicit outline, evidence, reasoning, conclusion structure.",
        how="Ask for four tagged stages: <SUMMARY> (restate the problem and outline the plan), <CAPTION> (the facts "
        "from the image or context the answer depends on), <REASONING> (step-by-step solution), and <CONCLUSION> "
        "(the final answer only).",
        how_reasoning="Ask for a staged deliverable: a one- or two-sentence approach, then the evidence relied on "
        "(the observations from the image or the facts from the context), then the conclusion. Do not ask for a "
        "written-out reasoning transcript: the model reasons internally.",
    ),
    Technique(
        id="verification",
        name="Chain-of-Verification",
        kind="prompt",
        papers=(COVE,),
        when="Factual questions, lists of entities, citations, or long-form factual writing where hallucination is "
        "a risk.",
        how="Ask the model to draft, then check each key factual claim (names, numbers, dates, citations) "
        "independently, correct or remove what does not hold up, and flag anything it cannot verify.",
    ),
    Technique(
        id="zero_shot_cot",
        name="Zero-shot chain-of-thought",
        kind="prompt",
        papers=(ZERO_SHOT_COT, COT),
        when="A reasoning task aimed at a model without built-in thinking.",
        how="Add an explicit step-by-step reasoning cue before the answer, and ask for the final answer on its own "
        "line.",
    ),
    Technique(
        id="re_reading",
        name="Re-reading",
        kind="prompt",
        papers=(RE2,),
        when="Long context precedes the question, or the problem statement is intricate.",
        how="Put the core question after the context, and restate it at the very end so the model reads it again "
        "right before answering.",
    ),
    Technique(
        id="specificity_format",
        name="Principled instructions",
        kind="prompt",
        papers=(PRINCIPLED, PROMPT_REPORT),
        when="Always: baseline prompt hygiene.",
        how="Be specific about goal, audience, scope, and success criteria. Separate instructions from data with XML "
        "tags or headings. Phrase directives as what to do. Specify the output format and length. Do not add tips, "
        "threats, or flattery.",
    ),
    Technique(
        id="self_consistency",
        name="Self-Consistency",
        kind="inference",
        papers=(SELF_CONSISTENCY,),
        when="Short, checkable answers (math, multiple choice) where a few independent samples are affordable.",
        how="Sample several independent answers and keep the final answer most of them agree on.",
    ),
    Technique(
        id="stage_search",
        name="LLaVA-CoT stage-wise search",
        kind="inference",
        papers=(LLAVA_COT,),
        when="Staged outputs from a small or open vision-language model.",
        how="Generate a few candidates for each stage, keep the best one, then continue to the next stage "
        "(LLaVA-CoT's stage-wise retracing search, SWIRES; earlier versions used stage-level beam search).",
    ),
    Technique(
        id="tree_of_thoughts",
        name="Tree of Thoughts",
        kind="inference",
        papers=(TOT,),
        when="Search-heavy puzzles and planning problems.",
        how="Explore several partial solutions, have the model evaluate each, and backtrack from dead ends.",
    ),
)

BY_ID: dict[str, Technique] = {t.id: t for t in TECHNIQUES}
PROMPT_TECHNIQUE_IDS: tuple[str, ...] = tuple(t.id for t in TECHNIQUES if t.kind == "prompt")


def catalog_text() -> str:
    """The prompt-technique catalog as plain text, stable across calls (used in the rewriter's system prompt)."""
    lines = []
    for t in TECHNIQUES:
        if t.kind != "prompt":
            continue
        lines.append(f"- {t.id}: {t.name} [{t.citation()}]")
        lines.append(f"  When: {t.when}")
        lines.append(f"  How: {t.how}")
        if t.how_reasoning:
            lines.append(f"  How, for reasoning targets: {t.how_reasoning}")
    return "\n".join(lines)

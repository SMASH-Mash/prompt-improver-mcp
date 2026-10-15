"""Heuristic task profiling and technique selection.

Profiling is lexical, so it is a set of hints rather than ground truth: the LLM backends receive it as
"suggestions" and may override it, and the host-mode draft says so too.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from prompt_improver.models import TaskProfile, TechniqueRef, Turn
from prompt_improver.techniques import BY_ID, PROMPT_TECHNIQUE_IDS, Target, Technique

_I = re.IGNORECASE

TASK_PATTERNS: dict[str, re.Pattern[str]] = {
    "visual": re.compile(
        r"\b(images?|pictures?|photos?|photographs?|screenshots?|diagrams?|charts?|plots?|figures?\b(?!\s+out)|"
        r"videos?|frames?|scans?|drawings?|infographics?|x-?rays?|scenes?)\b",
        _I,
    ),
    "math": re.compile(
        r"(\d\s*[-+*/^×÷=]\s*\d|\b(calculate|compute|solve|equations?|integral|derivative|probability|prove|proof|"
        r"percent(age)?|how (many|much)|average|median|ratio|sum of|formula|algebra|geometry|arithmetic)\b)",
        _I,
    ),
    "code": re.compile(
        r"(```|\b(code|function|method|bug|error|exception|traceback|stack ?trace|compile|refactor|implement|api|"
        r"endpoint|sql|regex|python|javascript|typescript|java|c\+\+|rust|golang|script|unit tests?|deploy|repo|git|"
        r"npm|pip|docker(file)?)\b|\b\w+\.(py|js|ts|tsx|java|cpp|rs|go|rb|cs|php)\b|\b\w+\(\))",
        _I,
    ),
    "factual": re.compile(
        r"\b(who (is|was|were|invented|wrote|founded)|when (did|was|were)|where (is|was)|(which|what) year|"
        r"list (of|all)|cite|citations?|references|sources|papers? (on|about)|history of|statistics|facts? about|"
        r"according to|biography)\b",
        _I,
    ),
    "knowledge": re.compile(
        r"\b(physics|chemistry|biology|law|legal|economics|theorem|principle|thermodynamics|mechanics|quantum|"
        r"explain why|why (does|do|is|are)|what causes)\b",
        _I,
    ),
    "multi_part": re.compile(
        r"\b(compare|contrast|pros and cons|trade-?offs?|step[- ]by[- ]step|design|architecture|analy[sz]e|"
        r"evaluate|and then)\b",
        _I,
    ),
    "planning": re.compile(
        r"\b(puzzle|game of 24|sudoku|crossword|schedule|itinerary|optimi[sz]e|route|allocate|constraints?)\b", _I
    ),
    "writing": re.compile(
        r"\b(write|draft|essay|story|poem|email|letter|blog|article|rewrite|tone|summari[sz]e|summary|paraphrase|"
        r"translate)\b",
        _I,
    ),
}
REASONING_TYPES = frozenset({"visual", "math", "knowledge", "multi_part", "planning"})

_HISTORY_REF = re.compile(
    r"\b(the above|above (code|answer|response|output|text)|previous(ly)?|earlier|last (one|answer|response|time|"
    r"version)|same (as|thing|way)|like before|as before|you (said|wrote|suggested|mentioned|gave)|your (answer|"
    r"response|code|suggestion|output|version)|(that|this) (code|answer|one|version|error|approach|function)|again)\b",
    _I,
)
_PRONOUN = re.compile(r"\b(it|this|that|these|those|them)\b", _I)
_DISSATISFIED = re.compile(
    r"(^\s*no\b|\bnot (what|quite|right|correct|working)\b|\b(wrong|incorrect|inaccurate)\b|"
    r"\b(doesn'?t|does not|didn'?t|did not|won'?t) (work|run|compile|help|fix|make sense)\b|"
    r"\bstill (fails?|failing|broken|wrong|not|the same|getting)\b|\btry again\b|\b(that'?s|this is) not\b|"
    r"\byou (missed|forgot|ignored|misunderstood)\b|\binstead\b|\bbroken\b)",
    _I,
)
_VAGUE = re.compile(r"\b(improve|better|fix|optimi[sz]e|help( me)?|make it|do (it|this|that)|something|stuff)\b", _I)
_ENUMERATED_LINE = re.compile(r"^\s*(\d+[.)]|[-*•])\s+", re.MULTILINE)

LONG_CONTEXT_CHARS = 8000


def profile_task(prompt: str, context: str, history: list[Turn], has_image: bool = False) -> TaskProfile:
    has_history = bool(history)
    words = len(prompt.split())
    refs = [m.group(0) for m in _HISTORY_REF.finditer(prompt)]
    if has_history and not refs and words <= 20:
        refs = [m.group(0) for m in _PRONOUN.finditer(prompt)]
    references_history = has_history and bool(refs)
    dissatisfied = any(t.role == "assistant" for t in history) and bool(_DISSATISFIED.search(prompt))

    # A follow-up like "fix it" carries its task type in the turn it refers to.
    typed_text = prompt
    if references_history or dissatisfied:
        earlier = next((t.content for t in reversed(history) if t.role == "user"), "")
        typed_text = f"{prompt}\n{earlier}"

    task_types: list[str] = []
    signals: list[str] = []
    for name, pattern in TASK_PATTERNS.items():
        match = pattern.search(typed_text)
        if match:
            task_types.append(name)
            signals.append(f"{name}: '{match.group(0).strip()}'")
    if has_image and "visual" not in task_types:
        task_types.insert(0, "visual")
        signals.append("visual: has_image=true")
    if "multi_part" not in task_types:
        questions = prompt.count("?")
        items = len(_ENUMERATED_LINE.findall(prompt))
        if questions >= 2 or items >= 2:
            task_types.append("multi_part")
            signals.append(f"multi_part: {questions} questions, {items} list items")
    if not task_types:
        task_types.append("general")

    if references_history:
        signals.append(f"refers to earlier turns: {', '.join(dict.fromkeys(r.lower() for r in refs))}")
    if dissatisfied:
        signals.append("corrects or retries a previous response")

    has_background = has_history or bool(context.strip())
    vague = bool(_VAGUE.search(prompt))
    if words < 5:
        ambiguity = "medium" if has_background else "high"
    elif words < 12 and vague:
        ambiguity = "medium" if has_background else "high"
    else:
        ambiguity = "low"

    context_chars = len(context) + sum(len(t.content) for t in history)
    return TaskProfile(
        task_types=task_types,
        signals=signals,
        has_history=has_history,
        references_history=references_history,
        history_references=list(dict.fromkeys(r.lower() for r in refs)) if references_history else [],
        dissatisfied_with_previous=dissatisfied,
        ambiguity=ambiguity,
        context_chars=context_chars,
        long_context=context_chars > LONG_CONTEXT_CHARS,
    )


@dataclass(frozen=True)
class Pick:
    technique: Technique
    why: str

    def ref(self) -> TechniqueRef:
        t = self.technique
        return TechniqueRef(id=t.id, name=t.name, citation=t.citation(), why=self.why)


def select_techniques(
    profile: TaskProfile, target: Target, forced: list[str] | None = None, limit: int = 7
) -> list[Pick]:
    """Choose prompt techniques for this task, in the order they should shape the rewrite."""
    if forced:
        unknown = [t for t in forced if t not in PROMPT_TECHNIQUE_IDS]
        if unknown:
            raise ValueError(f"Unknown technique id(s) {unknown}; valid ids: {', '.join(PROMPT_TECHNIQUE_IDS)}")
        return [Pick(BY_ID[t], "requested") for t in dict.fromkeys(forced)]

    types = set(profile.task_types)
    reasoning = types & REASONING_TYPES
    picks: list[Pick] = []

    def add(tid: str, why: str) -> None:
        if all(p.technique.id != tid for p in picks):
            picks.append(Pick(BY_ID[tid], why))

    if profile.has_history or profile.context_chars:
        add("context_distillation", "context or history supplied; keep only what the request depends on")
    if profile.references_history:
        add("self_contained_rewrite", f"refers to earlier turns ({', '.join(profile.history_references[:3])})")
    if profile.dissatisfied_with_previous:
        add("failure_feedback", "the prompt corrects or retries a previous response")
    if profile.ambiguity != "low":
        add("rephrase_expand", "the prompt is short or vague")
    if profile.ambiguity == "high":
        add("clarify", "missing details could change the answer substantially")
    if "visual" in types:
        add("visual_grounding", "the input includes an image or other visual")
    if "knowledge" in types:
        add("step_back", "a governing principle likely determines the answer")
    if "math" in types:
        add("plan_and_solve", "calculation or multi-step math")
    if types & {"multi_part", "planning"}:
        add("least_to_most", "several dependent parts")
    if "visual" in types or len(reasoning) >= 2:
        add("staged_reasoning", "multi-step reasoning benefits from summary, evidence, reasoning, conclusion stages")
    if "factual" in types:
        add("verification", "factual claims at risk of hallucination")
    if target == "standard" and reasoning and all(p.technique.id != "staged_reasoning" for p in picks):
        add("zero_shot_cot", "reasoning task for a model without built-in thinking")
    if profile.long_context:
        add("re_reading", "long context precedes the question")

    picks = picks[: limit - 1]
    add("specificity_format", "baseline: clear goal, delimiters, explicit output format")
    return picks


def inference_tips(profile: TaskProfile, picks: list[Pick], target: Target) -> list[str]:
    types = set(profile.task_types)
    chosen = {p.technique.id for p in picks}
    tips = []
    if types & {"math", "planning"}:
        t = BY_ID["self_consistency"]
        tips.append(f"{t.name}: {t.how} [{t.citation()}]")
    if "staged_reasoning" in chosen and target == "standard":
        t = BY_ID["stage_search"]
        tips.append(f"{t.name}: {t.how} [{t.citation()}]")
    if "planning" in types:
        t = BY_ID["tree_of_thoughts"]
        tips.append(f"{t.name}: {t.how} [{t.citation()}]")
    return tips

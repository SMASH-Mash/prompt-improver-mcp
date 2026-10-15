"""Offline approximation of System 2 Attention: keep the context segments relevant to the prompt.

S2A has an LLM regenerate the context. Without an LLM (host mode) we approximate that by scoring
segments with BM25 against the prompt, boosting recent turns, and always keeping the turns the prompt
points at. The result is a draft for the host model to refine, and the omitted count is reported.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from prompt_improver.models import Turn

_STOP = frozenset(
    "a an the and or but if then else of to in on at for with by from as is are was were be been being it this "
    "that these those i you he she we they me my your our their its do does did can could should would will just "
    "so not no yes what which who whom how why when where please".split()
)
_FENCE = re.compile(r"```.*?```", re.DOTALL)
LABELS = {"context": "context", "user": "earlier user message", "assistant": "earlier assistant response"}


@dataclass(frozen=True)
class Segment:
    source: str  # "context" | "user" | "assistant"
    turn: int  # history index, -1 for context
    order: int
    text: str


@dataclass(frozen=True)
class Distilled:
    segments: list[Segment]
    omitted_segments: list[Segment]

    @property
    def omitted(self) -> int:
        return len(self.omitted_segments)


def _terms(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9_]+", text.lower()) if len(w) > 1 and w not in _STOP]


def _paragraphs(text: str) -> list[str]:
    """Split on blank lines, keeping fenced code blocks whole."""
    blocks: list[str] = []
    pos = 0
    for fence in _FENCE.finditer(text):
        blocks.extend(re.split(r"\n\s*\n", text[pos : fence.start()]))
        blocks.append(fence.group(0))
        pos = fence.end()
    blocks.extend(re.split(r"\n\s*\n", text[pos:]))
    return [b.strip() for b in blocks if b.strip()]


def segment(context: str, history: list[Turn], whole_turn_chars: int = 1500) -> list[Segment]:
    segments: list[Segment] = []
    for para in _paragraphs(context):
        segments.append(Segment("context", -1, len(segments), para))
    for i, turn in enumerate(history):
        parts = [turn.content.strip()] if len(turn.content) <= whole_turn_chars else _paragraphs(turn.content)
        for part in parts:
            if part:
                segments.append(Segment(turn.role, i, len(segments), part))
    return segments


def distill(
    prompt: str,
    context: str,
    history: list[Turn],
    *,
    budget: int = 4000,
    keep_last_exchange: bool = False,
) -> Distilled:
    segments = segment(context, history)
    if not segments:
        return Distilled([], [])

    query = Counter(_terms(prompt))
    docs = [Counter(_terms(s.text)) for s in segments]
    n = len(docs)
    avgdl = sum(sum(d.values()) for d in docs) / n or 1.0
    df = Counter(t for d in docs for t in d)
    k1, b = 1.5, 0.75

    def bm25(doc: Counter[str]) -> float:
        dl = sum(doc.values())
        score = 0.0
        for term in query:
            tf = doc.get(term, 0)
            if tf:
                idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
                score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
        return score

    n_turns = max(len(history), 1)
    scores = []
    for seg, doc in zip(segments, docs):
        recency = (seg.turn + 1) / n_turns if seg.turn >= 0 else 0.0
        scores.append(bm25(doc) * (1 + 0.5 * recency))

    forced: set[int] = set()
    if keep_last_exchange:
        for role in ("assistant", "user"):
            turns = [s.turn for s in segments if s.source == role]
            if turns:
                forced |= {s.order for s in segments if s.turn == max(turns)}

    ranked = sorted(range(n), key=lambda i: (i not in forced, -scores[i]))
    chosen: list[Segment] = []
    used = 0
    for i in ranked:
        seg = segments[i]
        if i not in forced and scores[i] <= 0:
            break
        if used + len(seg.text) > budget:
            if i in forced and used < budget:
                keep = budget - used
                seg = Segment(seg.source, seg.turn, seg.order, f"{seg.text[:keep]}\n[... {len(seg.text) - keep} chars omitted]")
            else:
                continue
        chosen.append(seg)
        used += len(seg.text)
    chosen.sort(key=lambda s: s.order)
    kept = {s.order for s in chosen}
    return Distilled(chosen, [s for s in segments if s.order not in kept])

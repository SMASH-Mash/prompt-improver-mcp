"""Input and output models shared by the engine and the MCP tools."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from prompt_improver.techniques import Target

Backend = Literal["auto", "anthropic", "sampling", "host"]


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class TaskProfile(BaseModel):
    task_types: list[str] = Field(description="Detected task types, e.g. visual, math, code, factual.")
    signals: list[str] = Field(description="The evidence behind each detection.")
    has_history: bool
    references_history: bool = Field(description="The prompt points at earlier turns ('it', 'the above', ...).")
    history_references: list[str] = Field(description="The phrases that point at earlier turns.")
    dissatisfied_with_previous: bool = Field(description="The prompt corrects or retries a previous response.")
    ambiguity: Literal["low", "medium", "high"]
    context_chars: int
    long_context: bool


class TechniqueRef(BaseModel):
    id: str
    name: str
    citation: str
    why: str


class Analysis(BaseModel):
    profile: TaskProfile
    techniques: list[TechniqueRef]
    inference_tips: list[str] = Field(description="Inference-time techniques to consider; not part of the prompt text.")


class ImprovedPrompt(BaseModel):
    improved_prompt: str
    backend_used: Literal["anthropic", "sampling", "host"]
    target: Target
    profile: TaskProfile
    techniques: list[TechniqueRef]
    changes: list[str]
    assumptions: list[str]
    clarifying_questions: list[str]
    inference_tips: list[str]
    host_instructions: str | None = Field(
        default=None,
        description="Set only when backend_used is 'host': the improved_prompt is a template draft, and these are "
        "the steps the calling model should take to finish it.",
    )
    notes: list[str] = Field(default_factory=list)


class TechniqueInfo(BaseModel):
    id: str
    name: str
    kind: Literal["prompt", "inference"]
    when: str
    how: str
    how_for_reasoning_models: str | None
    papers: list[str]

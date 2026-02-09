"""
Pydantic models for API request/response validation.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Any


# === Simulation ===

class CreateSessionRequest(BaseModel):
    config: dict[str, Any] | None = None
    preset: str | None = None
    name: str | None = None


class RunRequest(BaseModel):
    generations: int | None = None


class StepRequest(BaseModel):
    n: int = 1


class SessionSummary(BaseModel):
    id: str
    name: str
    status: str
    current_generation: int
    max_generations: int
    population_size: int


class SessionResponse(BaseModel):
    id: str
    name: str
    status: str
    current_generation: int
    max_generations: int
    population_size: int
    config: dict[str, Any]


# === Agents ===

class AgentSummaryResponse(BaseModel):
    id: str
    name: str
    age: int
    generation: int
    birth_order: int
    processing_region: str
    suffering: float
    is_alive: bool
    partner_id: str | None
    is_outsider: bool
    dominant_voice: str | None
    latest_contribution: float


class AgentDetailResponse(AgentSummaryResponse):
    traits: dict[str, float]
    traits_at_birth: dict[str, float]
    trait_history: list[dict[str, float]]
    region_history: list[str]
    contribution_history: list[float]
    suffering_history: list[float]
    parent1_id: str | None
    parent2_id: str | None
    children_ids: list[str]
    relationship_status: str
    burnout_level: float
    personal_memories: list[dict[str, Any]]
    inherited_lore: list[dict[str, Any]]
    decision_history: list[dict[str, Any]]
    outsider_origin: str | None
    injection_generation: int | None


class PaginatedAgentList(BaseModel):
    agents: list[AgentSummaryResponse]
    total: int
    page: int
    page_size: int


# === Metrics ===

class SummaryResponse(BaseModel):
    total_generations: int
    final_population_size: int
    total_breakthroughs: int
    mean_contribution: float
    mean_suffering: float
    peak_population: int
    total_births: int
    total_deaths: int


class TimeSeriesResponse(BaseModel):
    field: str
    generations: list[int]
    values: list[Any]


# === Experiments ===

class PresetInfo(BaseModel):
    name: str
    config: dict[str, Any]


class ArchetypeInfo(BaseModel):
    name: str
    display_name: str
    description: str
    key_traits: list[str]
    use_case: str


class ArchetypeDetailResponse(ArchetypeInfo):
    trait_values: dict[str, float]


class CompareRequest(BaseModel):
    session_ids: list[str]


class ComparisonResponse(BaseModel):
    sessions: dict[str, SummaryResponse]
    config_diffs: dict[str, dict[str, list]]


class InjectRequest(BaseModel):
    session_id: str
    archetype: str | None = None
    custom_traits: dict[str, float] | None = None
    noise_sigma: float = 0.05
    name: str | None = None
    gender: str | None = None
    age: int | None = None
    injection_generation: int | None = None


# === Sensitivity ===

class SensitivityRequest(BaseModel):
    session_ids: list[str]
    target_metric: str = "mean_contribution"

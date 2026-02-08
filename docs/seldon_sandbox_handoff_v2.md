# The Seldon Sandbox: Claude Code Handoff Document v2.0

## Project Overview

The Seldon Sandbox is a **multi-generational societal simulation engine** designed as an experimentation platform. It models how personality traits, cognitive processing styles, and environmental pressures interact to produce emergent social structures across generations.

The deeper purpose: understanding how to compose teams and communities of agents with complementary traits — what mix of processing styles optimizes group output, how birth-order inheritance shapes community resilience, and what personality compositions make new settlements succeed or fail.

**Key Design Principles:**
1. **Everything is a slider** — No hardcoded assumptions; all parameters are tunable
2. **Track and visualize over time** — Every metric has a time series
3. **Compare runs** — A/B testing of parameter configurations
4. **Core is simple, complexity is opt-in** — Extension architecture for future features
5. **Decisions are mathematical and explainable** — Utility-based model with per-trait contribution analysis
6. **Memory shapes behavior** — Generational lore with fidelity decay

---

## Source Documents

### Primary Specifications
- **`seldon_sandbox_v3.0_architecture.md`** — Master architecture (READ THIS FIRST) with:
  - Configurable trait system (15/50/custom)
  - Utility-based decision model
  - Cognitive council (8 sub-agents)
  - RSH Five Regions processing model
  - Configurable birth order inheritance
  - Lore & memory system with fidelity decay
  - Outsider interface with ripple tracking
  - Extension architecture (geography, resources, migration, conflict, culture, technology)
  - Narrative visualization architecture (10 views)
  - Full ExperimentConfig specification

### Background Conversations
These conversations shaped the design. Read them for context when working on specific features:

- **`Gemini Conversation about Modeling a Society.docx`** — Origin story. Contains the 8-sub-agent cognitive architecture (Cortex, Seer, Oracle, House, Prudence, Hypothalamus, Amygdala, Conscience), compatibility scoring with detailed weights, infidelity model, the "Telephone Game" lore degradation mechanic, hexagonal world grid, and the "Outsider Interface" concept.

- **`ChatGPT Modeling of A Society.docx`** — Mathematical rigor. Contains the 50-trait taxonomy with functional groupings, the utility function formalization (`U(a|P,x) = P^T · W_a · x + b_a` with softmax), 11 archetype seed vectors with full 50-dimensional values (Da Vinci, Einstein, Montessori, Socrates, Curie, Fred Rogers, John Dewey, Dumbledore, Yoda, Ada Lovelace, Carl Sagan), emotional state inference, and the CreatureMind agent integration plan.

- **`Claude Conversation about modeling the society.docx`** — Simulation results. Contains 10-generation simulation data, migration modeling results (9.5-13.3% migration rates), attraction/mating model calibration, fertility constraint numbers, personality type emergence patterns, and population growth dynamics.

- **`Characteristics - experiential.docx`** — Echo Nexus AI mind project. Contains experiential encoding architecture, assertoric signals, qualitative similarity spaces, phenomenal memory, personality evolution through experience, and the 60/40 trait-utility/experiential-modulation decision split. Relevant for future LLM agent integration.

---

## Project Structure

```
seldon-sandbox/
├── CLAUDE.md                           # Claude Code guidance
├── README.md
├── pyproject.toml                      # Poetry project config
├── requirements.txt
│
├── src/
│   └── seldon/
│       ├── __init__.py
│       │
│       ├── core/                       # Core simulation components
│       │   ├── __init__.py
│       │   ├── config.py              # ExperimentConfig dataclass
│       │   ├── traits.py             # TraitSystem (configurable N traits)
│       │   ├── agent.py              # Agent dataclass
│       │   ├── inheritance.py        # InheritanceEngine (birth order rules)
│       │   ├── processing.py         # ProcessingClassifier (RSH Five Regions)
│       │   ├── drift.py              # TraitDriftEngine
│       │   ├── attraction.py         # AttractionModel
│       │   ├── decision.py           # DecisionModel (utility-based)
│       │   ├── council.py            # CognitiveCouncil (8 sub-agents)
│       │   └── engine.py             # SimulationEngine (main loop)
│       │
│       ├── social/                     # Social dynamics
│       │   ├── __init__.py
│       │   ├── relationships.py       # Pairing, dissolution, re-pairing
│       │   ├── fertility.py           # Reproduction constraints
│       │   └── lore.py               # LoreEngine, Memory, fidelity decay
│       │
│       ├── metrics/                    # Tracking and analysis
│       │   ├── __init__.py
│       │   ├── collector.py           # MetricsCollector
│       │   └── export.py             # Export for visualization
│       │
│       ├── experiment/                 # Experiment running
│       │   ├── __init__.py
│       │   ├── runner.py             # ExperimentRunner
│       │   ├── presets.py            # Preset configurations
│       │   ├── archetypes.py         # 11 archetype seed vectors
│       │   └── outsider.py           # OutsiderInterface + RippleTracker
│       │
│       ├── extensions/                 # Optional modules
│       │   ├── __init__.py
│       │   ├── base.py               # SimulationExtension ABC
│       │   ├── registry.py           # ExtensionRegistry
│       │   ├── geography.py
│       │   ├── resources.py
│       │   ├── migration.py          # Includes settlement viability
│       │   ├── conflict.py
│       │   ├── culture.py            # Meme propagation
│       │   └── technology.py
│       │
│       └── llm/                        # LLM integration (build last)
│           ├── __init__.py
│           └── interviewer.py
│
├── tests/
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_traits.py
│   ├── test_inheritance.py
│   ├── test_processing.py
│   ├── test_drift.py
│   ├── test_attraction.py
│   ├── test_decision.py
│   ├── test_council.py
│   ├── test_lore.py
│   ├── test_relationships.py
│   ├── test_outsider.py
│   ├── test_engine.py
│   └── test_metrics.py
│
├── frontend/                           # React web dashboard
│   ├── package.json
│   ├── src/
│   │   ├── views/
│   │   │   ├── Dashboard.tsx          # Mission Control
│   │   │   ├── PopulationOverview.tsx # Time series + heatmaps
│   │   │   ├── AgentExplorer.tsx      # Individual agent deep-dive
│   │   │   ├── SufferingContribution.tsx  # R4/R5 key visualization
│   │   │   ├── FamilyLineage.tsx      # Dynasty tracker
│   │   │   ├── SettlementDiag.tsx     # Settlement forensics
│   │   │   ├── NetworkView.tsx        # Attraction/social graphs
│   │   │   ├── LoreEvolution.tsx      # Memory decay visualization
│   │   │   ├── ExperimentCompare.tsx  # A/B test comparison
│   │   │   └── AnomalyDetector.tsx    # "Generation X was special"
│   │   ├── components/
│   │   │   ├── TraitRadar.tsx         # Radar chart for trait profiles
│   │   │   ├── RegionTimeline.tsx     # R1-R5 area chart
│   │   │   ├── AgentTimeline.tsx      # "Life of an agent" band
│   │   │   ├── ConfigPanel.tsx        # Slider-based config editor
│   │   │   └── ...
│   │   └── api/
│   │       └── simulation.ts          # API client to Python backend
│   └── ...
│
├── notebooks/
│   ├── 01_basic_simulation.ipynb
│   ├── 02_birth_order_experiments.ipynb
│   ├── 03_archetype_societies.ipynb
│   ├── 04_outsider_injection.ipynb
│   └── 05_settlement_composition.ipynb
│
├── examples/
│   ├── run_baseline.py
│   ├── run_ab_test.py
│   ├── run_archetype_society.py
│   ├── run_outsider_experiment.py
│   └── parameter_sweep.py
│
└── docs/
    ├── seldon_sandbox_v3.0_architecture.md
    ├── seldon_sandbox_handoff_v2.md
    └── [conversation .docx files]
```

---

## Implementation Order

### Phase 1: Foundation (Priority: P0)

Build these first, in order. Each component builds on the previous.

#### 1. `src/seldon/core/traits.py`
```python
# TraitSystem class — CONFIGURABLE trait count
# - Reads trait preset from config ('compact'=15, 'full'=50, 'custom')
# - Dynamic index constants (TraitSystem.CONSCIENTIOUSNESS, etc.)
# - Desirability vector for inheritance
# - Stability vector for drift
# - random_traits() generator
# - trait_name(index), trait_index(name) helpers
#
# CRITICAL: Never hardcode trait count. Use self.count everywhere.
# Shape of trait arrays is always (self.count,), not (15,) or (50,).
```

#### 2. `src/seldon/core/config.py`
```python
# ExperimentConfig dataclass — ALL tunable parameters
# - trait_preset ('compact', 'full', 'custom')
# - founding_population, founding_archetypes
# - birth_order_rules dict
# - region_thresholds, productive_weights
# - trait_drift_rate, region_effects
# - attraction_weights
# - decision_temperature, decision_weights, decision_biases
# - cognitive_council_enabled, cognitive_council_weights
# - relationship_config (dissolution, infidelity, LGBTQ, etc.)
# - fertility_config (age limits, spacing, mortality, societal pressure)
# - lore_decay_rate, lore_myth_threshold, lore_mutation_rate
# - outsider_injection_age, scheduled_injections
# - extensions_enabled, extensions dict
# - to_dict() and from_dict() for serialization
```

#### 3. `src/seldon/core/agent.py`
```python
# Agent dataclass — see architecture doc Part 4
# - traits: np.ndarray of shape (trait_system.count,)
# - Processing region, state, relationships, cognitive council state
# - History lists (trait, region, contribution, suffering)
# - personal_memories, inherited_lore
# - decision_history for explainability
# - social_bonds, infidelity_history
# - Outsider tracking fields
# - extension_data dict for extension hooks
```

#### 4. `src/seldon/core/processing.py`
```python
# ProcessingRegion enum (5 regions: UNDER_PROCESSING, OPTIMAL, DEEP, SACRIFICIAL, PATHOLOGICAL)
# ProcessingClassifier class
# - classify(agent, config) -> ProcessingRegion
# - Uses depth_drive trait + burnout + suffering
# - productive_potential calculation for R4 vs R5 distinction
# - ALL thresholds from config (not hardcoded!)
```

#### 5. `src/seldon/core/inheritance.py`
```python
# InheritanceEngine class
# - inherit(parent1, parent2, birth_order, population, rng) -> np.ndarray
# - Rule types: "worst", "weirdest", "best", "random_weighted", "average"
# - Uses TraitSystem.desirability for worst/best calculations
# - Uses population mean for weirdest calculations
# - Gaussian noise: epsilon ~ N(0, sigma^2)
# - Rules configurable per birth order position
# - Dead children count for birth order assignment
```

#### 6. `src/seldon/core/drift.py`
```python
# TraitDriftEngine class
# - drift_traits(agent, experiences) -> np.ndarray
# - apply_region_effects(agent) -> np.ndarray
# - Age-based stability (drift decreases with age)
# - Trait-specific stability from TraitSystem.stability
# - Region effects from config.region_effects
# - Always clamp result to [0, 1]
```

#### 7. `src/seldon/core/attraction.py`
```python
# AttractionModel class
# - calculate(agent1, agent2) -> float in [0, 1]
# - Components (all weights from config.attraction_weights):
#   - similarity: trait vector cosine similarity
#   - complementarity: how well traits complement each other
#   - universal_attractiveness: based on "universally desirable" traits
#   - social_proximity: shared social bonds
#   - age_compatibility: age difference penalty
#   - random_chemistry: stochastic attraction element
```

#### 8. `src/seldon/core/decision.py`
```python
# DecisionModel class — unified utility-based decisions
# - decide(agent, context, situation, actions) -> DecisionResult
# - U(action|P,x) = P^T · W_action · x + b_action
# - Softmax selection with configurable temperature
# - DecisionResult includes:
#   - chosen_action
#   - probabilities (all actions)
#   - trait_contributions (explainability)
# - DecisionContext enum: PAIRING, REPRODUCTION, MIGRATION, CONFLICT, etc.
#
# This replaces scattered threshold checks throughout the codebase.
# Every agent choice flows through this model.
```

#### 9. `src/seldon/core/engine.py`
```python
# SimulationEngine class — main loop
# - 7 phases per generation (see architecture doc Part 11):
#   1. Age & trait drift
#   2. Processing region updates
#   3. Contribution & breakthroughs
#   4. Relationship dynamics (dissolution, new pairings)
#   5. Reproduction with lore transmission
#   6. Lore evolution
#   7. Mortality (with partner widowing)
# - Extension hooks at: sim start, gen start/end, agent created,
#   modify attraction, modify mortality, modify decision
# - Outsider injection processing
# - Ripple effect tracking
```

### Phase 2: Intelligence & Social Layer (Priority: P1)

#### 10. `src/seldon/core/council.py`
```python
# CognitiveCouncil class — optional modulation layer
# - 8 sub-agents: Cortex, Seer, Oracle, House, Prudence, Hypothalamus, Amygdala, Conscience
# - Each sub-agent's influence = sum(trait_weights × agent_traits)
# - compute_council_modulation(agent, context) -> modulation vector
# - get_dominant_voice(agent) -> which sub-agent is strongest
# - Enabled/disabled via config.cognitive_council_enabled
# - When disabled, returns identity (no effect on decisions)
```

#### 11. `src/seldon/social/lore.py`
```python
# Memory dataclass — what agents know/believe
# MemoryType enum: PERSONAL, FAMILY, SOCIETAL, MYTH
# LoreEngine class:
# - transmit_to_child(parent, child, rng) — with fidelity decay
# - evolve_societal_lore(population, generation) — consensus, conflict, mutation
# - Fidelity decay: fidelity *= (1 - config.lore_decay_rate) per transmission
# - Below myth_threshold: becomes MYTH type, content may mutate
# - Emotional salience affects transmission probability
```

#### 12. `src/seldon/social/relationships.py`
```python
# RelationshipManager class
# - process_dissolutions(population, config) -> count
# - check_infidelity(agent, population, config) -> events
# - handle_partner_death(agent, deceased, population)
# - Dissolution based on compatibility score vs threshold
# - Infidelity modified by personality traits
# - Re-pairing cooldown after dissolution
```

#### 13. `src/seldon/social/fertility.py`
```python
# FertilityManager class
# - can_reproduce(agent1, agent2, config) -> bool
# - will_reproduce(agent1, agent2, config) -> bool (probability-based)
# - Age limits, birth spacing, maternal mortality
# - Societal pressure factor
# - Non-traditional family handling
```

#### 14. `src/seldon/metrics/collector.py`
```python
# GenerationMetrics dataclass — comprehensive per-generation snapshot
# MetricsCollector class
# - collect(generation, population, events) -> GenerationMetrics
# - get_time_series(metric_name) -> List[float]
# - export_for_visualization() -> dict (JSON-serializable)
# - Tracks: trait stats, region distribution, contribution, suffering,
#   demographics, birth order analysis, outsider diffusion, lore metrics
```

#### 15. `src/seldon/experiment/runner.py`
```python
# ExperimentRunner class
# - run_experiment(config) -> MetricsCollector
# - compare_experiments(names, metric) -> statistical comparison
# - run_ab_test(config_a, config_b, n_runs) -> significance test
# - run_parameter_sweep(base_config, param_name, values) -> results
```

#### 16. `src/seldon/experiment/presets.py`
```python
# Preset configuration functions:
# - config_baseline() — default parameters
# - config_no_birth_order() — all children get random_weighted
# - config_inverted_birth_order() — 1st=best, 3rd=worst
# - config_high_sacrificial() — lower threshold for R4 entry
# - config_no_recovery() — burnout doesn't heal
# - config_high_trait_drift() — rapid personality changes
# - config_opposites_attract_only() — complementarity-only attraction
# - config_archetype_society(archetype_mix) — found from archetypes
# - config_high_lore_decay() — rapid memory degradation
# - config_stable_lore() — minimal memory decay
```

#### 17. `src/seldon/experiment/archetypes.py`
```python
# Archetype definitions — 11 pre-built personality templates
# Each has:
# - full_vector: np.ndarray shape (50,) — complete 50-trait profile
# - compact_vector: np.ndarray shape (15,) — auto-projected 15-trait profile
# - name, description, key_traits summary
#
# Archetypes: da_vinci, einstein, montessori, socrates, curie,
#   fred_rogers, john_dewey, dumbledore, yoda, ada_lovelace, carl_sagan
#
# Source: ChatGPT conversation document (full 50-dim vectors provided there)
```

#### 18. `src/seldon/experiment/outsider.py`
```python
# OutsiderInterface class
# - inject_outsider(population, traits, generation, config) -> Agent
# - inject_archetype(population, archetype_name, generation, config) -> Agent
# - schedule_injection(config, generation, archetype, count) — pre-schedule
#
# RippleTracker class
# - track_injection(outsider, population) — snapshot at injection time
# - track_generation(outsider_id, generation, population) — ongoing tracking
# - get_diffusion_report(outsider_id) -> dict of metrics over time
# - find_descendants(outsider_id, population) -> List[Agent]
```

### Phase 3: Extensions Framework (Priority: P2)

#### 19. `src/seldon/extensions/base.py` + `registry.py`
```python
# SimulationExtension ABC — all hooks with default pass implementations
# - Lifecycle: on_simulation_start, on_generation_start/end, on_agent_created
# - Modifiers: modify_attraction, modify_mortality, modify_decision (NEW)
# - Metrics: get_metrics
# - Config: get_default_config (includes 'requires' for dependencies)
#
# ExtensionRegistry — with dependency resolution
# - register(), enable() (checks deps), disable(), get_enabled()
```

#### 20-25. Extension implementations
Build in this order (migration elevated for agent orchestration relevance):

1. **Geography** — Spatial dimension, hex grid, locations, carrying capacity
2. **Migration** — Push/pull factors, settlement founding, **composition viability analysis**
3. **Resources** — Scarcity dynamics, resource types, distribution
4. **Technology** — Breakthrough effects on capacity/mortality/fertility
5. **Culture/Memes** — Meme propagation, competition, evolution
6. **Conflict** — Personality-based triggers, resolution via decision model

### Phase 4: Frontend (Priority: P2)

React + Tailwind CSS + shadcn/ui. Recharts for charts, D3.js for networks/Sankey.

Build views in this order:
1. **Dashboard / Mission Control** — Config sliders, run controls, preset picker
2. **Population Overview** — Time series, trait heatmap, region stacked area
3. **Suffering vs. Contribution** — The key R4/R5 scatter plot
4. **Agent Explorer** — Individual deep-dive with "Life of an Agent" timeline
5. **Experiment Comparison** — Side-by-side runs with stat significance
6. **Family & Lineage** — Dynasty tracker, inheritance tree
7. **Settlement Diagnostics** — Composition analysis, failure forensics
8. **Network View** — Attraction/social graphs
9. **Lore Evolution** — Memory fidelity decay curves
10. **Anomaly Detector** — Flagging unusual generations

### Phase 5: LLM Integration (Priority: P3 — Build Last)

Agent interviews, narrative generation, decision narration.

---

## Key Implementation Notes

### 1. All Parameters Come From Config

**NEVER hardcode thresholds or weights.** Always read from `config`:

```python
# WRONG
if depth_drive > 0.8:
    return ProcessingRegion.SACRIFICIAL

# RIGHT
if depth_drive > config.region_thresholds['deep_to_extreme']:
    return ProcessingRegion.SACRIFICIAL
```

### 2. Trait Arrays Are N-Dimensional (Not Fixed at 15)

```python
# WRONG — hardcoded shape
agent.traits = np.array([0.5, 0.7, 0.3, ...])  # 15 values

# RIGHT — shape from trait system
trait_system = TraitSystem(config)
agent.traits = trait_system.random_traits(rng)  # shape (trait_system.count,)

# WRONG — hardcoded index
conscientiousness = agent.traits[1]

# RIGHT — named index from trait system
conscientiousness = agent.traits[trait_system.trait_index('conscientiousness')]
# or: agent.traits[trait_system.CONSCIENTIOUSNESS]  (dynamic attribute)
```

### 3. Decisions Go Through the Decision Model

```python
# WRONG — scattered threshold check
if agent.traits[AMBITION] > 0.7 and resources < 0.3:
    migrate = True

# RIGHT — utility-based decision
result = decision_model.decide(
    agent=agent,
    context=DecisionContext.MIGRATION,
    situation=np.array([resource_level, overcrowding, family_ties, ...]),
    actions=['stay', 'migrate', 'found_settlement'],
)
# result.chosen_action, result.probabilities, result.trait_contributions
```

### 4. Processing Regions Are Enums

```python
from enum import Enum

class ProcessingRegion(Enum):
    UNDER_PROCESSING = "under_processing"
    OPTIMAL = "optimal"
    DEEP = "deep"
    SACRIFICIAL = "sacrificial"
    PATHOLOGICAL = "pathological"
```

### 5. History Tracking Is Essential

Every agent appends to history lists each generation. These drive all visualizations:

```python
agent.trait_history.append(agent.traits.copy())
agent.region_history.append(agent.processing_region)
agent.contribution_history.append(contribution)
agent.suffering_history.append(agent.suffering)
```

### 6. Extension Hooks Pattern

Extensions can modify behavior at defined points without touching core code:

```python
# Core code calls hooks
for ext in self.extensions.get_enabled():
    ext.on_generation_start(gen, population, self.config)

# Extensions modify values
for ext in self.extensions.get_enabled():
    mortality_rate = ext.modify_mortality(agent, mortality_rate, self.config)

# NEW in v3.0: Extensions can modify decision utilities
for ext in self.extensions.get_enabled():
    utilities = ext.modify_decision(agent, context, utilities, self.config)
```

### 7. Birth Order Rules

Rules are configurable per birth order position. The default is:
- 1st: `"worst"` — less desirable parent value per trait (using desirability map)
- 2nd: `"weirdest"` — whichever parent is farther from population mean
- 3rd: `"best"` — more desirable parent value per trait
- 4th+: `"random_weighted"` — random mix of both parents

Dead children count for birth order. Identical twins share position; fraternal twins get distinct positions.

### 8. RSH Five Regions

| Region | Description | Key Characteristic |
|--------|-------------|-------------------|
| R1: Under-Processing | Quick, shallow decisions | Low depth, low contribution |
| R2: Optimal | Balanced, efficient | Peak efficiency, sustainable |
| R3: Deep | Thorough, costly | High contribution, recoverable |
| R4: Sacrificial | Obsessive, productive suffering | Breakthroughs despite high cost |
| R5: Pathological | Obsessive, unproductive suffering | High cost, no output |

The R4 vs R5 distinction (productive potential) is determined by creativity, resilience, and burnout level — all configurable weights.

### 9. Cognitive Council is Optional

The council modulates decisions but is **disabled by default** for performance. Enable via `config.cognitive_council_enabled = True`. When disabled, the decision model works identically — the council step is identity.

### 10. Lore Fidelity Decay

Each memory transmission: `fidelity *= (1 - config.lore_decay_rate)`

At `config.lore_decay_rate = 0.05`, a memory reaches myth status (~0.3 fidelity) after approximately 24 transmissions. With one transmission per generation, this is ~24 generations — matching the Gemini conversation's "~20 generations" estimate.

---

## Testing Strategy

### Unit Tests for Each Component

```python
# test_traits.py
def test_compact_trait_system_has_15_traits():
    config = ExperimentConfig(trait_preset='compact')
    ts = TraitSystem(config)
    assert ts.count == 15

def test_full_trait_system_has_50_traits():
    config = ExperimentConfig(trait_preset='full')
    ts = TraitSystem(config)
    assert ts.count == 50

def test_dynamic_index_constants():
    config = ExperimentConfig(trait_preset='compact')
    ts = TraitSystem(config)
    assert hasattr(ts, 'CONSCIENTIOUSNESS')
    assert ts.trait_name(ts.CONSCIENTIOUSNESS) == 'conscientiousness'

# test_inheritance.py
def test_worst_inheritance_positive_trait():
    """First-born gets min of positive-desirability trait."""
    config = ExperimentConfig()
    ts = TraitSystem(config)
    engine = InheritanceEngine(config)

    p1_traits = np.full(ts.count, 0.8)
    p2_traits = np.full(ts.count, 0.3)

    child = engine.inherit(p1, p2, birth_order=1, population=[])
    assert child[ts.CONSCIENTIOUSNESS] == pytest.approx(0.3, abs=0.1)

def test_weirdest_inheritance():
    """Second-born gets trait farther from population mean."""
    config = ExperimentConfig()
    ts = TraitSystem(config)
    engine = InheritanceEngine(config)

    pop_mean = np.full(ts.count, 0.5)
    p1_traits = np.full(ts.count, 0.9)  # Far from mean
    p2_traits = np.full(ts.count, 0.6)  # Close to mean

    child = engine.inherit(p1, p2, birth_order=2, population=mock_pop)
    # Should get p1's value (farther from mean)
    assert child[ts.OPENNESS] == pytest.approx(0.9, abs=0.1)

def test_birth_order_rules_are_configurable():
    """Can invert birth order rules."""
    config = ExperimentConfig(birth_order_rules={1: 'best', 3: 'worst'})
    engine = InheritanceEngine(config)
    # First-born should now get best traits
    child = engine.inherit(p1, p2, birth_order=1, population=[])
    assert child[ts.CONSCIENTIOUSNESS] == pytest.approx(0.8, abs=0.1)

# test_decision.py
def test_decision_model_returns_valid_probabilities():
    """Probabilities sum to 1."""
    result = decision_model.decide(agent, DecisionContext.PAIRING, situation, actions)
    assert pytest.approx(sum(result.probabilities.values()), abs=1e-6) == 1.0

def test_decision_explainability():
    """DecisionResult includes per-trait contributions."""
    result = decision_model.decide(agent, DecisionContext.MIGRATION, situation, actions)
    assert len(result.trait_contributions) == trait_system.count

# test_lore.py
def test_lore_fidelity_decays_on_transmission():
    memory = Memory(fidelity=1.0, ...)
    transmitted = lore_engine._copy_with_decay(memory, gen=5, rng)
    assert transmitted.fidelity < 1.0
    assert transmitted.fidelity == pytest.approx(1.0 - config.lore_decay_rate)

def test_low_fidelity_becomes_myth():
    memory = Memory(fidelity=0.25, ...)  # Below myth_threshold
    transmitted = lore_engine._copy_with_decay(memory, gen=30, rng)
    assert transmitted.memory_type == MemoryType.MYTH

# test_outsider.py
def test_outsider_injection():
    outsider = interface.inject_outsider(population, traits, gen=10, config)
    assert outsider.is_outsider
    assert outsider.injection_generation == 10
    assert outsider in population

def test_ripple_tracker_finds_descendants():
    # Create outsider, pair, reproduce
    descendants = tracker._find_descendants(outsider.id, population)
    assert len(descendants) > 0
```

### Integration Tests

```python
def test_full_generation_runs():
    config = ExperimentConfig(initial_population=20)
    engine = SimulationEngine(config, ExtensionRegistry())
    metrics = engine.run(generations=1)
    assert metrics.history[0].population_size > 0

def test_trait_drift_over_generations():
    config = ExperimentConfig(trait_drift_rate=0.1, initial_population=50)
    engine = SimulationEngine(config, ExtensionRegistry())
    metrics = engine.run(generations=10)
    # Traits should have changed
    gen0_means = metrics.history[0].trait_means
    gen9_means = metrics.history[9].trait_means
    assert not np.allclose(gen0_means, gen9_means, atol=0.01)

def test_outsider_ripple_effect():
    config = ExperimentConfig(
        initial_population=50,
        scheduled_injections=[{'generation': 5, 'archetype': 'einstein', 'count': 1}],
    )
    engine = SimulationEngine(config, ExtensionRegistry())
    metrics = engine.run(generations=20)
    # Einstein traits should have diffused into population
    # (check via ripple tracker metrics)

def test_archetype_founding_population():
    config = ExperimentConfig(
        founding_population='archetypes',
        founding_archetypes=[('curie', 5), ('fred_rogers', 5)],
    )
    engine = SimulationEngine(config, ExtensionRegistry())
    metrics = engine.run(generations=5)
    assert metrics.history[0].population_size == 10

def test_compact_and_full_traits_produce_results():
    """Both trait presets run without errors."""
    for preset in ['compact', 'full']:
        config = ExperimentConfig(trait_preset=preset, initial_population=20)
        engine = SimulationEngine(config, ExtensionRegistry())
        metrics = engine.run(generations=3)
        assert len(metrics.history) == 3
```

### Property-Based Tests

```python
from hypothesis import given, strategies as st

@given(st.floats(0, 1), st.floats(0, 1))
def test_attraction_is_bounded(trait1, trait2):
    """Attraction scores should always be in [0, 1]."""
    # ... construct agents with these trait values ...
    score = attraction_model.calculate(a1, a2)
    assert 0.0 <= score <= 1.0

@given(st.integers(1, 10))
def test_inheritance_always_produces_valid_traits(birth_order):
    """Child traits should always be in [0, 1]."""
    child_traits = engine.inherit(p1, p2, birth_order, population)
    assert np.all(child_traits >= 0.0) and np.all(child_traits <= 1.0)

@given(st.floats(0.01, 0.5))
def test_lore_fidelity_monotonically_decreases(decay_rate):
    """Fidelity should never increase through transmission."""
    # ... transmit N times, check each fidelity <= previous
```

---

## Example Usage

### Basic Simulation

```python
from seldon.core.config import ExperimentConfig
from seldon.core.engine import SimulationEngine
from seldon.extensions.registry import ExtensionRegistry

config = ExperimentConfig(
    experiment_name="my_first_run",
    initial_population=100,
    generations_to_run=50,
    birth_order_rules={1: "worst", 2: "weirdest", 3: "best"},
    trait_drift_rate=0.02,
)

engine = SimulationEngine(config, ExtensionRegistry())
metrics = engine.run(config.generations_to_run)

print(f"Final population: {metrics.history[-1].population_size}")
print(f"Total breakthroughs: {sum(m.breakthrough_count for m in metrics.history)}")
print(f"Final R4 proportion: {metrics.history[-1].region_proportions['sacrificial']:.1%}")
```

### Archetype Society Experiment

```python
config = ExperimentConfig(
    experiment_name="einstein_curie_society",
    founding_population='archetypes',
    founding_archetypes=[
        ('einstein', 6),
        ('curie', 5),
    ],
    founding_trait_noise=0.1,
    generations_to_run=100,
)

engine = SimulationEngine(config, ExtensionRegistry())
metrics = engine.run(config.generations_to_run)
# Analyze: does a society of deep thinkers produce more breakthroughs?
# Or does it collapse from too much suffering?
```

### Outsider Injection Experiment

```python
config = ExperimentConfig(
    experiment_name="disruptor",
    initial_population=100,
    generations_to_run=50,
    scheduled_injections=[
        {'generation': 25, 'archetype': 'da_vinci', 'count': 1},
    ],
)

engine = SimulationEngine(config, ExtensionRegistry())
metrics = engine.run(config.generations_to_run)

# Analyze ripple effect
report = engine.ripple_tracker.get_diffusion_report('outsider_25_...')
print(f"Descendants after 25 gens: {report['descendant_count']}")
print(f"Trait diffusion score: {report['trait_diffusion']:.3f}")
```

### A/B Test: Birth Order Effects

```python
from seldon.experiment.runner import ExperimentRunner
from seldon.experiment.presets import config_baseline, config_no_birth_order

runner = ExperimentRunner()

runner.run_experiment(config_baseline())
runner.run_experiment(config_no_birth_order())

comparison = runner.compare_experiments(
    ['baseline', 'no_birth_order'],
    'breakthrough_count'
)
print(f"Breakthrough difference: {comparison['effect_size']:.2f}")
print(f"Significant: {comparison['p_value'] < 0.05}")
```

### Settlement Composition Experiment (with extensions)

```python
from seldon.extensions.geography import GeographyExtension
from seldon.extensions.migration import MigrationExtension

config = ExperimentConfig(
    experiment_name="settlement_test",
    initial_population=200,
    generations_to_run=100,
)

registry = ExtensionRegistry()
registry.register(GeographyExtension())
registry.register(MigrationExtension())
registry.enable('geography')
registry.enable('migration')

config.enable_extension(GeographyExtension())
config.enable_extension(MigrationExtension())
config.configure_extension('geography', map_size=(15, 15), starting_locations=3)
config.configure_extension('migration', new_settlement_enabled=True)

engine = SimulationEngine(config, registry)
metrics = engine.run(config.generations_to_run)
# Analyze which settlements succeeded and why
```

---

## Dependencies

```
# requirements.txt

# Core
numpy>=1.24.0
dataclasses-json>=0.6.0      # Config serialization
scipy>=1.10.0                 # Statistical tests for A/B comparisons

# Testing
pytest>=7.0.0
pytest-cov>=4.0.0
hypothesis>=6.0.0             # Property-based testing

# Visualization (Python side)
matplotlib>=3.7.0
pandas>=2.0.0
plotly>=5.0.0

# Notebooks
jupyter>=1.0.0

# API server (for frontend)
fastapi>=0.100.0
uvicorn>=0.23.0
pydantic>=2.0.0

# Frontend (in frontend/package.json)
# react, tailwindcss, @shadcn/ui, recharts, d3
```

---

## Questions Resolved in v3.0

| Question from v1.0 | Resolution |
|---|---|
| **Pairing duration** — permanent or dissolvable? | Configurable. Default: dissolution enabled at low compatibility threshold. |
| **Multiple children per generation?** | `max_children_per_generation: 1` in config. |
| **Death of partner — re-pair?** | Yes. `reparing_after_death: True` in config with optional cooldown. |
| **Contribution calculation?** | Base from region_effects, modified by creativity/resilience, through decision model. |
| **Breakthrough mechanics?** | Increments counter + tech_level, creates personal memory, triggers lore event. |

## New Questions for Implementation

1. **Frontend data transport** — WebSocket for live simulation updates, or REST polling?
   - Suggestion: Start with REST; add WebSocket for real-time stepping later.

2. **Persistence** — Save/load simulation state for pausing and resuming?
   - Suggestion: JSON serialization of full state; implement after core works.

3. **Parallel runs** — Run A/B tests in parallel processes?
   - Suggestion: Use multiprocessing for parameter sweeps; single-thread for core dev.

4. **Archetype vector sources** — Extract exact 50-dim vectors from ChatGPT doc?
   - The ChatGPT conversation contains the complete vectors. Parse during archetypes.py implementation.

---

## Context from Project Creator

This project was designed by Kevin Keller of Tucuxi Inc. Key context:

1. **RSH Background** — The Reasoning Saturation Hypothesis is Kevin's research on cognitive processing patterns. The Five Regions model comes from this work.

2. **Agent Orchestration** — The deeper purpose is understanding how to compose teams/communities of agents with complementary traits. Settlement viability analysis is the most directly relevant feature.

3. **Experimentation Focus** — This is NOT meant to simulate "reality" — it's a what-if engine. Birth order rules are hypotheses to test, not facts.

4. **No Equilibrium Targets** — We want to see what emerges from different parameter settings.

5. **Visualization Matters** — Being able to see time series, trait distributions, region populations, and agent stories is essential.

6. **Mathematical Foundation** — The utility-based decision model provides a rigorous, explainable foundation for all agent behavior.

7. **Memory Shapes Culture** — The lore system creates emergent mythology and cultural effects without hardcoding cultural rules.

---

*Handoff Document Version: 2.0*
*Architecture Version: 3.0*
*Previous Version: 1.0 (referencing architecture v2.3)*
*Last Updated: February 2025*

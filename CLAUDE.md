# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Identity

The Seldon Sandbox (named after Hari Seldon from Asimov's *Foundation*) is a **multi-generational societal simulation engine** — an experimentation platform for exploring how personality traits, cognitive processing styles, birth order inheritance rules, and environmental pressures interact to produce emergent social structures across generations. It is designed by Kevin Keller of Tucuxi Inc.

This is NOT a "reality simulator." It is a **what-if engine** — an experimental sandbox for testing hypotheses about community composition, agent orchestration, and emergent group dynamics. The deeper purpose: understanding how to compose teams and communities of agents with complementary traits.

## Core Design Principles

1. **Everything is a slider** — No hardcoded assumptions. All thresholds, weights, rates, and rules come from `ExperimentConfig`. Never hardcode a magic number.
2. **Track and visualize over time** — Every metric has a time series. History tracking on agents is not optional.
3. **Compare runs** — A/B testing and parameter sweeps are first-class features.
4. **Core is simple, complexity is opt-in** — Extension architecture for optional modules.
5. **Decisions are mathematical and explainable** — Utility-based model: `U(a|P,x) = P^T · W_a · x + b_a` with softmax. Per-trait contribution analysis for every decision.
6. **Memory shapes behavior** — Generational lore with fidelity decay creates emergent mythology.
7. **LLM for interviews only** — The simulation runs on pure math. LLMs are only used later for agent interviews and narrative generation.

## Architecture Overview

```
EXPERIMENT RUNNER (A/B testing, parameter sweeps, archetype experiments)
    └── EXPERIMENT CONFIG (all parameters as tunable sliders)
            ├── CORE ENGINE
            │   ├── TraitSystem (configurable: 15 compact / 50 full / custom)
            │   ├── Agent (dataclass with history, lore, decision tracking)
            │   ├── DecisionModel (utility-based, all choices flow through this)
            │   ├── CognitiveCouncil (optional 8-voice modulation layer)
            │   ├── InheritanceEngine (birth order: worst/weirdest/best/random/average)
            │   ├── ProcessingClassifier (RSH Five Regions)
            │   ├── TraitDriftEngine (experience + age-based trait changes)
            │   ├── AttractionModel (similarity, complementarity, chemistry)
            │   └── SimulationEngine (generation loop with extension hooks)
            ├── SOCIAL
            │   ├── RelationshipManager (pairing, dissolution, infidelity)
            │   ├── FertilityManager (constraints, societal pressure)
            │   └── LoreEngine (memory transmission, fidelity decay, myths)
            ├── EXPERIMENT
            │   ├── ExperimentRunner, Presets
            │   ├── Archetypes (11 seed vectors: Da Vinci, Einstein, Curie, etc.)
            │   └── OutsiderInterface + RippleTracker
            ├── EXTENSIONS (optional, via ExtensionRegistry)
            │   ├── Geography, Migration (settlement composition analysis)
            │   ├── Resources, Technology
            │   ├── Culture/Memes, Conflict
            │   └── SimulationExtension ABC with lifecycle + modifier hooks
            └── METRICS & VISUALIZATION
                ├── MetricsCollector (per-generation stats)
                └── 10 narrative visualization views (React dashboard)
```

### Generation Loop Phases (7 phases in order)
1. **Age & Trait Drift** — Agents age, traits shift via drift engine and region effects
2. **Processing Region Updates** — Reclassify agents into RSH Five Regions; update dominant cognitive voice
3. **Contribution & Breakthroughs** — Calculate output, detect breakthroughs, create memories
4. **Relationship Dynamics** — Process dissolutions, form new pairs via decision model
5. **Reproduction** — Paired agents produce children via inheritance engine; transmit lore
6. **Lore Evolution** — Societal-level memory consensus, conflict, mutation
7. **Mortality** — Age/burnout/extension-modified death checks; handle widowing

Extension hooks fire at: simulation start, generation start/end, agent created, modify attraction, modify mortality, modify decision utilities, get metrics.

## The RSH Five Regions Model

From Kevin's Reasoning Saturation Hypothesis paper. This is central to the simulation:

| Region | Key Behavior | Output |
|--------|-------------|--------|
| R1: Under-Processing | Quick, shallow decisions | Low contribution |
| R2: Optimal | Balanced, efficient processing | Peak sustainable output |
| R3: Deep | Thorough but costly processing | High contribution, recoverable |
| R4: Sacrificial | Obsessive, productive suffering | Breakthroughs despite high cost (Van Gogh, Curie) |
| R5: Pathological | Obsessive, unproductive suffering | Pure loss — OCD loops, rumination |

**The R4 vs R5 distinction is critical**: both involve suffering, but R4 produces breakthroughs while R5 is pure loss. The `productive_potential_threshold` determines the boundary, calculated from creativity, resilience, and burnout level.

## Birth Order Inheritance Rules

The foundational hypothesis:
- **1st-born → "worst"**: Less desirable trait value per trait (using desirability map)
- **2nd-born → "weirdest"**: Whichever parent's trait is farther from population mean
- **3rd-born → "best"**: More desirable trait value per trait
- **4th+ → "random_weighted"**: Random mix of both parents
- Gaussian noise added for environmental/developmental variance
- Dead children count for birth-order assignment
- Identical twins share position; fraternal twins get distinct assignments
- Rules are configurable per position — testing inverted/disabled/custom rules is the whole point

## Key Technical Conventions

### Trait Arrays Are N-Dimensional (Not Fixed)
```python
# WRONG — hardcoded shape
agent.traits = np.array([0.5, 0.7, 0.3, ...])  # assumes 15

# RIGHT — shape from trait system
trait_system = TraitSystem(config)
agent.traits = trait_system.random_traits(rng)  # shape (trait_system.count,)

# WRONG — hardcoded index
conscientiousness = agent.traits[1]

# RIGHT — named index
conscientiousness = agent.traits[trait_system.CONSCIENTIOUSNESS]
# or: agent.traits[trait_system.trait_index('conscientiousness')]
```

### Decisions Go Through the Decision Model
```python
# WRONG — scattered threshold check
if agent.traits[AMBITION] > 0.7 and resources < 0.3:
    migrate = True

# RIGHT — utility-based decision
result = decision_model.decide(
    agent, DecisionContext.MIGRATION, situation_vector,
    actions=['stay', 'migrate', 'found_settlement']
)
# result.chosen_action, result.probabilities, result.trait_contributions
```

### Config-Driven Parameters
```python
# WRONG — hardcoded threshold
if depth_drive > 0.8:
    return ProcessingRegion.SACRIFICIAL

# RIGHT — from config
if depth_drive > config.region_thresholds['deep_to_extreme']:
    return ProcessingRegion.SACRIFICIAL
```

### History Tracking Is Essential
Every agent appends to history lists each generation. These drive all visualizations:
```python
agent.trait_history.append(agent.traits.copy())
agent.region_history.append(agent.processing_region)
agent.contribution_history.append(contribution)
agent.suffering_history.append(agent.suffering)
```

## Project Structure

```
seldon-sandbox/
├── src/seldon/
│   ├── core/          # TraitSystem, Agent, Config, Engine, Inheritance,
│   │                  #   Processing, Drift, Attraction, DecisionModel, CognitiveCouncil
│   ├── social/        # RelationshipManager, FertilityManager, LoreEngine
│   ├── metrics/       # MetricsCollector, export for visualization
│   ├── experiment/    # ExperimentRunner, Presets, Archetypes, OutsiderInterface
│   ├── extensions/    # SimulationExtension ABC, Registry, Geography, Migration,
│   │                  #   Resources, Technology, Culture, Conflict
│   └── llm/           # Agent interviews, narrative generation (build last)
├── tests/             # Unit + integration + property-based (hypothesis) tests
├── frontend/          # React web dashboard (10 visualization views)
├── notebooks/         # Jupyter exploration notebooks
├── examples/          # run_baseline.py, run_ab_test.py, run_archetype_society.py, etc.
└── docs/              # Architecture v3.0, handoff v2.0, conversation transcripts
```

## Build & Run Commands

```bash
# Install dependencies
pip install -e ".[dev]"
# or: poetry install

# Run all tests
pytest tests/

# Run a single test file
pytest tests/test_inheritance.py

# Run a specific test
pytest tests/test_inheritance.py::test_worst_inheritance_positive_trait -v

# Run with coverage
pytest --cov=seldon tests/

# Run a basic simulation
python examples/run_baseline.py

# Run an A/B comparison
python examples/run_ab_test.py

# Run an archetype society experiment
python examples/run_archetype_society.py

# Start the frontend dashboard (when implemented)
cd frontend && npm install && npm run dev
```

## Dependencies

Core: `numpy`, `scipy`, `dataclasses-json`
Testing: `pytest`, `pytest-cov`, `hypothesis`
Visualization: `matplotlib`, `pandas`, `plotly`
API: `fastapi`, `uvicorn`, `pydantic`
Frontend: React, Tailwind CSS, shadcn/ui, Recharts, D3.js
Notebooks: `jupyter`

## Implementation Priority

**Phase 1 (P0 — Core Engine):** TraitSystem → ExperimentConfig → Agent → ProcessingClassifier → InheritanceEngine → TraitDriftEngine → AttractionModel → DecisionModel → SimulationEngine

**Phase 2 (P1 — Intelligence & Social):** CognitiveCouncil → LoreEngine → RelationshipManager → FertilityManager → MetricsCollector → ExperimentRunner → Presets → Archetypes → OutsiderInterface + RippleTracker

**Phase 3 (P1 — Visualization):** Dashboard → Population Overview → Suffering vs. Contribution → Agent Explorer → Experiment Comparison → Family & Lineage

**Phase 4 (P2 — Extensions):** Extension ABC + Registry → Geography → Migration (settlement composition) → Resources → Technology → Culture/Memes → Conflict

**Phase 5 (P2 — Advanced Viz):** Settlement Diagnostics → Network View → Lore Evolution → Anomaly Detection → Parameter Sensitivity

**Phase 6 (P3 — LLM):** Agent interviews, narrative generation, decision narration

## Source Documentation

All project context lives in `docs/`:
- **`seldon_sandbox_v3.0_architecture.md`** — Master architecture spec (v3.0). READ THIS FIRST.
- **`seldon_sandbox_handoff_v2.md`** — Implementation guide v2.0: project structure, build order, code patterns, testing strategy, example usage
- `seldon_sandbox_v2.3_architecture.md` — Previous architecture (superseded by v3.0)
- `seldon_sandbox_handoff.md` — Previous handoff (superseded by v2.0)
- `Gemini Conversation about Modeling a Society.docx` — Origin story; 8-agent cognitive architecture, compatibility scoring, infidelity model, "Telephone Game" lore degradation, hexagonal grid, Outsider Interface concept
- `ChatGPT Modeling of A Society.docx` — Mathematical rigor; 50-trait taxonomy, utility function (`U(a|P,x) = P^T · W_a · x + b_a`), 11 archetype seed vectors with full 50-dim values, emotional state inference
- `Claude Conversation about modeling the society.docx` — Simulation results; 10-gen data, migration modeling, attraction calibration, fertility constraints, population dynamics
- `Characteristics - experiential.docx` — Echo Nexus AI mind; experiential encoding, assertoric signals, phenomenal memory, personality evolution (relevant for future LLM integration)

## Key Experiments the System Should Support

1. **Birth order hypothesis**: Does 1st=worst, 2nd=weirdest, 3rd=best produce emergent structures? What about inverted rules?
2. **Optimal processing mix**: What R1-R5 proportion maximizes societal contribution?
3. **Settlement composition**: What personality mix makes a new settlement succeed or fail?
4. **Outsider disruption**: How quickly do foreign traits propagate? Improve or destabilize?
5. **Cultural meme effects**: Does "tortured genius" maintain the sacrificial population?
6. **Lore degradation**: How do distorted memories affect collective behavior?
7. **Archetype societies**: What emerges from a society of Einsteins? Of Curies + Fred Rogers?
8. **Resource scarcity pressure**: How does scarcity change processing region distributions?

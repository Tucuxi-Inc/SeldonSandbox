# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Identity

The Seldon Sandbox (named after Hari Seldon from Asimov's *Foundation*) is a **multi-generational societal simulation engine** — an experimentation platform for exploring how personality traits, cognitive processing styles, birth order inheritance rules, and environmental pressures interact to produce emergent social structures across generations. It is designed by Kevin Keller of Tucuxi Inc.

This is NOT a "reality simulator." It is a **what-if engine** — an experimental sandbox for testing hypotheses about community composition, agent orchestration, and emergent group dynamics. The deeper purpose: understanding how to compose teams and communities of agents with complementary traits.

## Core Design Principles

1. **Everything is a slider** — No hardcoded assumptions. All thresholds, weights, rates, and rules come from `ExperimentConfig`. Never hardcode a magic number.
2. **Track and visualize over time** — Every metric has a time series. History tracking on agents is not optional.
3. **Compare runs** — A/B testing, parameter sweeps, and session cloning (fork) are first-class features.
4. **Core is simple, complexity is opt-in** — Extension architecture for optional modules.
5. **Decisions are mathematical and explainable** — Utility-based model: `U(a|P,x) = P^T · W_a · x + b_a` with softmax. Per-trait contribution analysis for every decision.
6. **Memory shapes behavior** — Generational lore with fidelity decay creates emergent mythology.
7. **LLM for narratives only** — The simulation runs on pure math. LLMs are used for biographies, chronicles, interviews, and explanations — never for determining what happens.
8. **Every death tells a story** — Detailed cause-of-death tracking with mortality factor breakdown.

## Architecture Overview

```
EXPERIMENT RUNNER (A/B testing, parameter sweeps, archetype experiments)
    └── EXPERIMENT CONFIG (all parameters as tunable sliders)
            ├── CORE ENGINE
            │   ├── TraitSystem (configurable: 15 compact / 50 full / custom)
            │   ├── Agent (dataclass with traits, history, lore, decisions, genome)
            │   ├── DecisionModel (utility-based, all choices flow through this)
            │   ├── CognitiveCouncil (optional 8-voice modulation layer)
            │   ├── InheritanceEngine (birth order: worst/weirdest/best/random/average)
            │   ├── ProcessingClassifier (RSH Five Regions)
            │   ├── TraitDriftEngine (experience + age-based trait changes)
            │   ├── AttractionModel (similarity, complementarity, chemistry)
            │   ├── SimulationEngine (9-phase generation loop with extension hooks)
            │   ├── TickEngine (12 ticks/year: needs, gathering, movement)
            │   ├── HexGrid (axial coordinates, 10 terrain types, A* pathfinding)
            │   ├── NeedsSystem (6 survival needs, terrain/season decay)
            │   ├── GatheringSystem (8 activities, personality-modulated yields)
            │   ├── TickActivityLog (per-tick agent activity capture for World View)
            │   ├── ExperientialEngine (6-dim felt-quality, recall, phenomenal quality)
            │   ├── GeneticModel (allele pairs, crossover, mutation, expression)
            │   ├── EpigeneticModel (environmental markers, transgenerational inheritance)
            │   └── GeneticAttribution (lineage tracking, trait-gene correlation)
            ├── SOCIAL
            │   ├── RelationshipManager (pairing, dissolution, infidelity)
            │   ├── FertilityManager (constraints, societal pressure)
            │   ├── LoreEngine (memory transmission, fidelity decay, myths)
            │   ├── SocialHierarchyManager (status, influence, role assignment)
            │   ├── MentorshipManager (matching, skill transfer, chains)
            │   ├── MarriageManager (courtship, formalization, divorce, political marriages)
            │   ├── ClanManager (founders, ancestry, honor, rivalries)
            │   ├── InstitutionManager (councils, guilds, elections, prestige)
            │   ├── CommunityManager (detection, cohesion, factions)
            │   └── BeliefSystem (formation, propagation, conflict, accuracy)
            ├── EXPERIMENT
            │   ├── ExperimentRunner, Presets
            │   ├── Archetypes (11 seed vectors: Da Vinci, Einstein, Curie, etc.)
            │   └── OutsiderInterface + RippleTracker
            ├── EXTENSIONS (optional, via ExtensionRegistry — 12 modules)
            │   ├── Geography, Migration (settlement composition analysis)
            │   ├── Resources, Technology
            │   ├── Culture/Memes, Conflict
            │   ├── Social Dynamics (hierarchy + mentorship + marriage + clans + institutions)
            │   ├── Diplomacy (alliances, rivalries, cultural exchange)
            │   ├── Economics (production, trade, wealth, occupations, governance)
            │   ├── Environment (seasons, climate, drought, plague, disease)
            │   ├── Epistemology (belief formation, propagation, accuracy dynamics)
            │   ├── Inner Life (experiential mind, phenomenal quality, experiential drift)
            │   └── SimulationExtension ABC with lifecycle + modifier hooks
            ├── LLM + NARRATIVE (never affects simulation)
            │   ├── ClaudeClient (Anthropic API) + OllamaClient (local, Docker-aware)
            │   ├── AgentInterviewer (in-character conversations, current + historical mode)
            │   ├── NarrativeGenerator (prose summaries)
            │   ├── DecisionNarrator (psychological analysis)
            │   ├── BiographyGenerator (structured + LLM-enriched agent biographies)
            │   └── EventExtractor (chronicle: breakthroughs, deaths, suffering, outsiders)
            └── METRICS, API & PERSISTENCE
                ├── MetricsCollector (per-generation stats)
                ├── SessionManager (in-memory sessions with SQLite persistence + cloning)
                ├── SessionStore (SQLite CRUD, zlib-compressed state blobs)
                ├── FastAPI REST API (18 routers, 70+ endpoints)
                └── React Dashboard (25 views, real-time updates)
```

### Generation Loop Phases (9 phases in order)
1. **Age & Trait Drift** — Agents age, traits shift via drift engine and region effects
2. **Epigenetic Updates** — Environmental markers activate/deactivate based on conditions (Phase 1.5, when genetics enabled)
3. **Processing Region Updates** — Reclassify agents into RSH Five Regions; update dominant cognitive voice
4. **Contribution & Breakthroughs** — Calculate output, detect breakthroughs, create memories
5. **Relationship Dynamics** — Process dissolutions, form new pairs via decision model
6. **Reproduction** — Paired agents produce children via inheritance engine (optionally with allele-based genetics); transmit lore
7. **Lore Evolution** — Societal-level memory consensus, conflict, mutation
8. **Mortality** — Age/burnout/extension-modified death checks with cause-of-death tracking; handle widowing
9. **Extension Hooks** — Extensions fire at 8 lifecycle points: simulation start, generation start/end, agent created, modify attraction, modify mortality, modify decision utilities, collect metrics

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
│   ├── core/          # TraitSystem, Agent, Config, Engine, TickEngine, HexGrid,
│   │                  #   MapGenerators, NeedsSystem, GatheringSystem, Inheritance,
│   │                  #   Processing, Drift, Attraction, DecisionModel, CognitiveCouncil,
│   │                  #   ExperientialEngine, GeneticModel, EpigeneticModel, GeneticAttribution
│   ├── social/        # RelationshipManager, FertilityManager, LoreEngine,
│   │                  #   SocialHierarchyManager, MentorshipManager, MarriageManager,
│   │                  #   ClanManager, InstitutionManager, CommunityManager, BeliefSystem
│   ├── metrics/       # MetricsCollector, export for visualization
│   ├── experiment/    # ExperimentRunner, Presets, Archetypes, OutsiderInterface
│   ├── extensions/    # SimulationExtension ABC, Registry, 12 extension modules
│   │                  #   (geography, migration, resources, technology, culture,
│   │                  #    conflict, social_dynamics, diplomacy, economics, environment,
│   │                  #    epistemology, inner_life)
│   ├── llm/           # ClaudeClient, OllamaClient, Interviewer, Narrator,
│   │                  #   BiographyGenerator, EventExtractor (Chronicle), Prompts
│   └── api/           # FastAPI app, SessionManager, SessionStore (SQLite persistence),
│                      #   Serializers, 17 routers (simulation, agents, metrics,
│                      #   experiments, settlements, network, advanced, llm, social,
│                      #   communities, economics, environment, genetics, beliefs,
│                      #   inner_life, hex_grid, chronicle)
├── tests/             # 1009 tests (pytest) + conftest.py for DB isolation
├── frontend/          # React + TypeScript + Tailwind v4 + Recharts + D3
│   └── src/components/views/  # 25 dashboard views across 7 sections
├── examples/          # run_baseline.py, run_phase2_demo.py
└── docs/              # Architecture v3.0, handoff v2.0, conversation transcripts
```

## Build & Run Commands

```bash
# Install dependencies (with API + dev extras)
pip install -e ".[api,dev]"

# Run all tests (1009 tests)
pytest tests/

# Run a single test file
pytest tests/test_inheritance.py

# Run a specific test
pytest tests/test_inheritance.py::test_worst_inheritance_positive_trait -v

# Run with coverage
pytest --cov=seldon tests/

# Run a basic simulation
python examples/run_baseline.py

# Start the backend API
uvicorn seldon.api.app:app --host 0.0.0.0 --port 8006 --reload

# Start the frontend dashboard (in a separate terminal)
cd frontend && npm install && npm run dev

# Or use Docker for the full stack
docker compose up --build
```

## Dependencies

Core: `numpy`, `scipy`, `dataclasses-json`
Testing: `pytest`, `pytest-cov`, `hypothesis`
Visualization: `matplotlib`, `pandas`, `plotly`
API: `fastapi`, `uvicorn`, `pydantic`, `python-dotenv`, `anthropic`
Persistence: `sqlite3` (stdlib — no extra install)
Frontend: React, TypeScript, Tailwind CSS v4, Recharts, D3.js, Zustand, Lucide icons
Notebooks: `jupyter`

## Implementation Status

All phases complete. Current state of the system:

**Phase 1 (Core Engine):** TraitSystem, ExperimentConfig, Agent, ProcessingClassifier, InheritanceEngine, TraitDriftEngine, AttractionModel, DecisionModel, SimulationEngine — **COMPLETE**

**Phase 2 (Intelligence & Social):** CognitiveCouncil, LoreEngine, RelationshipManager, FertilityManager, MetricsCollector, ExperimentRunner, Presets, Archetypes, OutsiderInterface + RippleTracker — **COMPLETE**

**Phase 3 (Visualization):** FastAPI backend, React dashboard, Population, Suffering, Agent Explorer, Experiments, Family & Lineage — **COMPLETE**

**Phase 4 (Extensions):** Extension ABC + Registry, Geography, Migration, Resources, Technology, Culture/Memes, Conflict — **COMPLETE**

**Phase 5 (Advanced Viz):** Settlement Diagnostics, Network View, Lore Evolution, Anomaly Detection, Parameter Sensitivity — **COMPLETE**

**Phase 6 (LLM):** Dual-provider client (Anthropic + Ollama), agent interviews, narrative generation, decision narration — **COMPLETE**

**Phase 7 (Social Dynamics):** SocialHierarchyManager, MentorshipManager, SocialDynamicsExtension, social API router — **COMPLETE**

**Phase 8 (Genetics):** GeneticModel, EpigeneticModel, GeneticAttribution, inheritance integration, engine Phase 1.5 — **COMPLETE**

**Phases 9-11 (Community, Economics, Environment):** CommunityManager, DiplomacyExtension, EconomicsExtension, EnvironmentExtension — **COMPLETE**

**Phase 12 (Frontend Controls):** Dashboard controls for all 10 extensions, outsider builder, 6 new views, genetics API router — **COMPLETE**

**Persistence:** SQLite session persistence with auto-save, lazy loading, compressed state blobs — **COMPLETE**

**Phase A (Tick Engine + Hex Grid + Needs):** TickEngine (12 ticks/year adapter), HexGrid (axial coordinates, 10 terrain types, A* pathfinding, map generators), NeedsSystem (6 survival needs), GatheringSystem (8 activities) — **COMPLETE**

**Phase B (Hex Grid Integration):** Agent placement, movement decisions, vision-range interaction filtering, cluster detection, hex grid persistence — **COMPLETE**

**Phase C (Economy Deepening):** Settlement dataclass, tick-level production/consumption/skills, inter-settlement trade, governance with taxation/poverty relief — **COMPLETE**

**Phase D (Rich Social Systems):** MarriageManager, ClanManager, InstitutionManager, all wired into SocialDynamicsExtension — **COMPLETE**

**Phase E (Belief Systems):** BeliefSystem, EpistemologyExtension, beliefs API router, 4 epistemology types, 6 belief domains — **COMPLETE**

**Phase F (Inner Life):** ExperientialEngine, InnerLifeExtension, inner-life API router, 6-dim felt-quality vectors, phenomenal quality — **COMPLETE**

**Phase G (Beliefs + Inner Life Views):** BeliefsView, InnerLifeView, frontend wiring — **COMPLETE**

**Narrative Overhaul:** Death tracking with cause-of-death breakdown, EventExtractor (chronicle), BiographyGenerator, chronicle API router, BiographyView, ChronicleView, AgentComparisonView, HexMapView, historical interview mode, session cloning (fork) — **COMPLETE**

**World View:** Sim City-like animated hex grid with tick-by-tick stepping, D3/SVG agent animation, playback controls, activity icons, needs visualization, event toasts, connection lines, movement trails, LLM thought bubbles, keyboard shortcuts — **COMPLETE**

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

## Key Experiments the System Supports

1. **Birth order hypothesis**: Does 1st=worst, 2nd=weirdest, 3rd=best produce emergent structures? What about inverted rules?
2. **Optimal processing mix**: What R1-R5 proportion maximizes societal contribution?
3. **Settlement composition**: What personality mix makes a new settlement succeed or fail?
4. **Outsider disruption**: How quickly do foreign traits propagate? Improve or destabilize?
5. **Cultural meme effects**: Does "tortured genius" maintain the sacrificial population?
6. **Lore degradation**: How do distorted memories affect collective behavior?
7. **Archetype societies**: What emerges from a society of Einsteins? Of Curies + Fred Rogers?
8. **Resource scarcity pressure**: How does scarcity change processing region distributions?
9. **Genetic drift vs selection**: Do allele frequencies shift toward optimal or random distributions?
10. **Epigenetic adaptation**: Do trauma markers persist and help future generations cope?
11. **Social hierarchy emergence**: What personality mixes produce stable vs. chaotic hierarchies?
12. **Economic inequality**: How does trade network topology affect wealth distribution?
13. **Belief system dynamics**: Do empirical beliefs outcompete sacred ones? How does accuracy evolve?
14. **Experiential divergence**: Do agents with similar traits but different experiences develop different personalities?
15. **Spatial clustering**: How does hex grid distance affect community formation and genetic diversity?

## Persistence & Data Model

Sessions are persisted to SQLite (`data/seldon.db` by default, configurable via `SELDON_DB_PATH`).

Key details:
- `SessionManager(db_path=)`: `None` = in-memory only, string = SQLite path
- Auto-save after every mutation (create, step, run_full, reset, delete)
- Lazy load: `_session_index` holds metadata; full state deserialized on `get_session()`
- State blob: zlib-compressed JSON with all_agents, living_agent_ids, metrics_history, next_agent_id, previous_regions
- Hex grid state persisted when tick engine active
- Extension state NOT persisted — rebuilt via `_build_extensions()` + `on_simulation_start()` on load
- RNG reseeded on load: `seed + current_generation`
- `conftest.py` sets `SELDON_DB_PATH` to tmpdir for test isolation
- Docker: `seldon-data` named volume at `/app/data`

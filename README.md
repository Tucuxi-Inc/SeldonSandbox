# The Seldon Sandbox

**A multi-generational societal simulation engine for exploring how personality traits, cognitive processing styles, inheritance rules, and environmental pressures produce emergent social structures across generations.**

Named after Hari Seldon from Asimov's *Foundation*, the Seldon Sandbox is a what-if engine -- an experimental platform for testing hypotheses about community composition, agent orchestration, and emergent group dynamics. The deeper purpose: understanding how to compose teams and communities of agents with complementary traits.

Built by [Kevin Keller](https://github.com/kkeller-tucuxi) of [Tucuxi Inc](https://tucuxi.com).

---

## Table of Contents

- [Core Ideas](#core-ideas)
- [Quick Start](#quick-start)
- [Running with Docker](#running-with-docker)
- [Running Locally (Development)](#running-locally-development)
- [The Dashboard](#the-dashboard)
- [Architecture](#architecture)
- [The RSH Five Regions Model](#the-rsh-five-regions-model)
- [Birth Order Inheritance](#birth-order-inheritance)
- [Trait System](#trait-system)
- [Extensions](#extensions)
- [LLM Integration](#llm-integration)
- [Experiment Presets](#experiment-presets)
- [Archetypes](#archetypes)
- [Running Experiments from Python](#running-experiments-from-python)
- [API Reference](#api-reference)
- [Tests](#tests)
- [Project Structure](#project-structure)
- [Design Principles](#design-principles)

---

## Core Ideas

The simulation models a population of agents, each defined by an N-dimensional personality trait vector (15 traits in compact mode, 50 in full mode). Every generation, agents:

1. **Age and drift** -- traits shift over time based on experience and processing region effects
2. **Get classified** into one of five cognitive processing regions (from the RSH model)
3. **Contribute** to society -- output depends on their processing region, creativity, and resilience
4. **Form and dissolve** relationships based on attraction (similarity, complementarity, chemistry)
5. **Reproduce** -- children inherit traits via birth-order rules (1st=worst, 2nd=weirdest, 3rd=best)
6. **Transmit memories and lore** -- stories degrade over generations, creating emergent myths
7. **Die** -- from age, burnout, or environmental pressure

All of this is driven by pure math -- utility functions, softmax decisions, configurable thresholds. No randomness is hidden; everything flows through a seeded RNG for full reproducibility. LLMs are used only after the fact, for interviewing agents and narrating events.

---

## Quick Start

### Docker (recommended)

```bash
git clone https://github.com/Tucuxi-Inc/SeldonSandbox.git
cd SeldonSandbox
docker compose up --build
```

Open [http://localhost:3006](http://localhost:3006) in your browser. The dashboard is ready to use.

### Local Development

```bash
git clone https://github.com/Tucuxi-Inc/SeldonSandbox.git
cd SeldonSandbox

# Backend
pip install -e ".[api,dev]"
uvicorn seldon.api.app:app --host 0.0.0.0 --port 8006

# Frontend (new terminal)
cd frontend
npm install
npm run dev
```

Open [http://localhost:3006](http://localhost:3006).

### CLI only (no dashboard)

```bash
pip install -e .
python examples/run_baseline.py
```

---

## Running with Docker

The project ships with Docker Compose for one-command deployment.

```bash
docker compose up --build
```

| Service | Port | Purpose |
|---------|------|---------|
| `backend` | 8006 | FastAPI simulation engine + REST API |
| `frontend` | 3006 | React dashboard (nginx serves built assets, proxies `/api` to backend) |

### Environment variables

Create a `.env` file in the project root (already gitignored):

```env
# Optional: enables Anthropic Claude for agent interviews/narratives
ANTHROPIC_API_KEY=sk-ant-...
```

The API key is passed into the Docker container automatically. Without it, the simulation runs normally -- LLM features just show an "unavailable" banner.

### Ollama (local LLM)

If you prefer running LLM features without an API key, install [Ollama](https://ollama.com) on your host machine and pull a model:

```bash
ollama pull llama3.2
```

The Docker container automatically connects to Ollama on the host via `host.docker.internal`. You can also set `OLLAMA_HOST` to point to a custom Ollama instance. In the dashboard, go to the Interview view's Settings tab and select "Ollama" as the provider.

---

## Running Locally (Development)

### Prerequisites

- Python 3.11+
- Node.js 20+
- (Optional) Ollama for local LLM
- (Optional) `ANTHROPIC_API_KEY` for Claude-powered interviews

### Backend

```bash
# Install with all development dependencies
pip install -e ".[all]"

# Start the API server
uvicorn seldon.api.app:app --host 0.0.0.0 --port 8006 --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The Vite dev server on port 3006 automatically proxies `/api` requests to the backend on port 8006.

### Run tests

```bash
pytest tests/
```

381 tests covering the core engine, social systems, extensions, API endpoints, and LLM integration. All LLM tests use mocked clients -- no real API calls.

---

## The Dashboard

The web dashboard at [http://localhost:3006](http://localhost:3006) provides 12 interactive views:

### Core Views

| View | What It Shows |
|------|--------------|
| **Mission Control** | Create sessions, configure parameters, toggle extensions, step through generations. The command center. |
| **Population** | Population dynamics over time -- births, deaths, growth rate, region distribution as stacked areas. |
| **Suffering & Contribution** | The central tension: scatter plots of suffering vs. contribution by processing region. Shows who produces at what cost. |
| **Agent Explorer** | Searchable, filterable list of all agents (living and dead). Click any agent for full trait profile, history charts, memories, and decision log. |
| **Experiments** | Side-by-side comparison of multiple sessions. Overlay time series. See how parameter changes affect outcomes. |
| **Family & Lineage** | Interactive family tree for any agent. Trace trait inheritance up through ancestors and down through descendants. |

### Advanced Views

| View | What It Shows |
|------|--------------|
| **Settlements** | When geography/migration extensions are enabled: settlement map, population by location, viability scores, migration timeline. |
| **Social Network** | D3 force-directed graph of all social bonds, partnerships, and parent-child relationships. Color-coded by processing region. |
| **Lore Evolution** | How societal memories change over time. Fidelity decay, myth formation, memory type distribution. |
| **Anomalies** | Statistical anomaly detection via z-scores. Flags unusual spikes in deaths, breakthroughs, suffering, or population shifts. |
| **Sensitivity** | Compare multiple sessions to see which parameters have the most impact. Tornado diagrams and correlation analysis. |

### LLM Views

| View | What It Shows |
|------|--------------|
| **Agent Interview** | Four-tab interface: (1) Chat with any agent in-character, (2) Generate prose narratives for any generation, (3) Get psychological explanations of agent decisions, (4) Configure LLM provider (Anthropic or Ollama). |

---

## Architecture

```
EXPERIMENT RUNNER (A/B testing, parameter sweeps, archetype experiments)
    |
    v
EXPERIMENT CONFIG (all parameters as tunable sliders)
    |
    +-- CORE ENGINE
    |   +-- TraitSystem ........... configurable: 15 compact / 50 full / custom
    |   +-- Agent ................. dataclass with traits, history, lore, decisions
    |   +-- DecisionModel ......... utility-based: U(a|P,x) = P^T * W_a * x + b_a
    |   +-- CognitiveCouncil ...... optional 8-voice processing modulation
    |   +-- InheritanceEngine ..... birth order rules (worst/weirdest/best/random)
    |   +-- ProcessingClassifier .. RSH Five Regions assignment
    |   +-- TraitDriftEngine ...... experience + age-based trait changes
    |   +-- AttractionModel ....... similarity, complementarity, chemistry
    |   +-- SimulationEngine ...... 7-phase generation loop with extension hooks
    |
    +-- SOCIAL
    |   +-- RelationshipManager ... pairing, dissolution, infidelity, widowing
    |   +-- FertilityManager ...... birth spacing, maternal/child mortality, pressure
    |   +-- LoreEngine ............ memory transmission, fidelity decay, myth formation
    |
    +-- EXTENSIONS (optional, via ExtensionRegistry)
    |   +-- Geography ............. hexagonal grid, settlements, terrain
    |   +-- Migration ............. settlement viability, group migration
    |   +-- Resources ............. resource production, distribution, scarcity
    |   +-- Technology ............ tech advancement, tool access
    |   +-- Culture ............... cultural memes, transmission, dominance
    |   +-- Conflict .............. inter-group tension, resolution
    |
    +-- LLM (narrative layer, never affects simulation)
    |   +-- ClaudeClient .......... Anthropic API wrapper
    |   +-- OllamaClient .......... Local model wrapper (Docker-aware)
    |   +-- AgentInterviewer ...... in-character conversations
    |   +-- NarrativeGenerator .... prose generation summaries
    |   +-- DecisionNarrator ...... psychological decision analysis
    |
    +-- METRICS & API
        +-- MetricsCollector ...... per-generation statistics
        +-- FastAPI REST API ...... 8 routers, 30+ endpoints
        +-- React Dashboard ....... 12 views, real-time updates
```

### Generation Loop (7 phases per generation)

1. **Age & Trait Drift** -- Agents age, traits shift based on drift rate and processing region effects
2. **Processing Region Updates** -- Reclassify into R1-R5; update dominant cognitive voice
3. **Contribution & Breakthroughs** -- Calculate output, detect breakthroughs, create memories
4. **Relationship Dynamics** -- Process dissolutions, form new pairs via the decision model
5. **Reproduction** -- Paired agents produce children; traits inherited via birth order rules; lore transmitted
6. **Lore Evolution** -- Societal-level memory consensus, conflict, mutation, myth formation
7. **Mortality** -- Age/burnout/extension-modified death checks; handle widowing

Extensions hook into this loop at 8 points: simulation start, generation start/end, agent created, modify attraction, modify mortality, modify decision utilities, and get metrics.

---

## The RSH Five Regions Model

Based on Kevin Keller's Reasoning Saturation Hypothesis. Every agent is classified into one of five cognitive processing regions based on their `depth_drive` trait and other factors:

| Region | Processing Style | Contribution | Cost |
|--------|-----------------|-------------|------|
| **R1: Under-Processing** | Quick, shallow decisions | Low | Low suffering |
| **R2: Optimal** | Balanced, efficient | Peak sustainable output | Moderate |
| **R3: Deep** | Thorough but costly | High | Recoverable strain |
| **R4: Sacrificial** | Obsessive, productive suffering | Breakthroughs | Burnout risk (think Van Gogh, Curie) |
| **R5: Pathological** | Obsessive, unproductive | None | Pure loss -- OCD loops, rumination |

**The R4 vs R5 distinction is critical**: both involve suffering, but R4 produces breakthroughs while R5 produces nothing. The `productive_potential_threshold` (configurable) determines the boundary, calculated from creativity, resilience, and burnout level.

One key experiment: What ratio of R1-R5 agents maximizes total societal contribution? Too many R4s burn out; too few means no breakthroughs.

---

## Birth Order Inheritance

The foundational hypothesis that drives emergent population dynamics:

| Birth Position | Rule | What It Means |
|---------------|------|---------------|
| 1st child | **worst** | For each trait, inherits the *less desirable* value from the two parents |
| 2nd child | **weirdest** | For each trait, inherits whichever parent's value is farther from the population mean |
| 3rd child | **best** | For each trait, inherits the *more desirable* value from the two parents |
| 4th+ | **random_weighted** | Random blend of both parents |

Gaussian noise (configurable via `inheritance_noise_sigma`) adds developmental variance. Dead children count for birth-order assignment. Twins are handled: identical twins share a position, fraternal twins get distinct assignments.

These rules are fully configurable -- testing inverted, disabled, or custom rules is the whole point. The `inverted_birth_order` preset flips 1st=best and 3rd=worst to test the opposite hypothesis.

---

## Trait System

Agents have N-dimensional personality trait vectors with values in [0, 1]. Two built-in presets:

### Compact Mode (15 traits -- default)

`openness`, `conscientiousness`, `extraversion`, `agreeableness`, `neuroticism`, `creativity`, `resilience`, `ambition`, `empathy`, `dominance`, `trust`, `risk_taking`, `adaptability`, `self_control`, `depth_drive`

### Full Mode (50 traits)

Includes everything in compact mode plus 35 additional traits covering cognitive processing, social dynamics, creative capacity, and more. See `src/seldon/core/traits.py` for the complete list.

### Custom Traits

You can define entirely custom trait sets:

```python
config = ExperimentConfig(
    trait_preset="custom",
    custom_traits=[
        {"name": "analytical_thinking", "desirability": 0.7, "stability": 0.8},
        {"name": "social_intuition", "desirability": 0.6, "stability": 0.7},
        # ...
    ]
)
```

Every trait has a `desirability` score (used by birth order rules to determine "worst" vs "best") and a `stability` score (how resistant it is to drift).

---

## Extensions

Extensions are optional modules that add environmental complexity. Enable them from the Mission Control dashboard or in config:

```python
config = ExperimentConfig(
    extensions_enabled=["geography", "migration", "resources"],
    extensions={
        "geography": {"grid_size": 10},
        "resources": {"base_production": 1.0},
    }
)
```

| Extension | What It Adds | Dependencies |
|-----------|-------------|--------------|
| **Geography** | Hexagonal grid, settlements with carrying capacity, terrain types | None |
| **Migration** | Settlement viability scoring, group migration, new settlement founding | Geography |
| **Resources** | Resource production, distribution, scarcity pressure | None |
| **Technology** | Technology level advancement, tool access, skill development | None |
| **Culture** | Cultural memes, transmission between agents, meme dominance | None |
| **Conflict** | Inter-group tension, conflict resolution, raiding | None |

Extensions hook into the generation loop at 8 points (simulation start, generation start/end, agent created, modify attraction/mortality/decisions, collect metrics). They can modify how agents make decisions, who they're attracted to, and how likely they are to die -- but the core simulation math stays the same.

---

## LLM Integration

The simulation runs on pure math. LLMs are used only for narrative features:

### Agent Interviews

Chat with any agent in character. The LLM receives the agent's full personality profile, memories, lore, and recent decisions, then responds as that character in first person.

### Generation Narratives

Generate prose summaries of any generation. The LLM weaves population statistics, notable agents, breakthroughs, and societal lore into a narrative.

### Decision Explanations

Select any decision from an agent's history and get a psychological analysis of why they chose what they did, based on their trait contributions and utility scores.

### Provider Options

| Provider | Setup | Best For |
|----------|-------|----------|
| **Anthropic (Claude)** | Set `ANTHROPIC_API_KEY` env var | Highest quality narratives |
| **Ollama (Local)** | Install Ollama, pull a model | Free, private, no API key needed |

Switch providers in the Interview view's Settings tab. Ollama auto-detects local models.

---

## Experiment Presets

Pre-configured experiment templates accessible from Mission Control or programmatically:

| Preset | What It Tests |
|--------|--------------|
| `baseline` | Default parameters -- the control group |
| `no_birth_order` | All children inherit via averaging (no birth order effects) |
| `inverted_birth_order` | 1st=best, 3rd=worst -- the opposite hypothesis |
| `high_sacrificial` | Lower R4 threshold -- more agents enter sacrificial processing |
| `no_recovery` | Agents cannot recover from deep/sacrificial/pathological regions |
| `high_trait_drift` | 5x normal trait drift -- personalities change fast |
| `opposites_attract` | High complementarity weight in pairing |
| `archetype_society` | Small population seed for archetype injection experiments |
| `high_lore_decay` | Memories become myths quickly |
| `stable_lore` | Memories persist for many generations |

---

## Archetypes

11 pre-defined personality seed vectors for outsider injection experiments:

| Archetype | Personality | Use Case |
|-----------|-------------|----------|
| **Da Vinci** | Polymath -- boundless curiosity, creativity across domains | Creative/innovative societies |
| **Einstein** | Deep thinker -- intense focus, theoretical breakthroughs | Deep-processing dominated societies |
| **Montessori** | Educator -- empathetic, structured, patient nurturing | Nurturing/educational communities |
| **Socrates** | Questioner -- relentless inquiry, challenges assumptions | Critical thinking societies |
| **Curie** | Sacrificial genius -- relentless pursuit at personal cost | R4 processing outcomes |
| **Fred Rogers** | Compassionate connector -- unconditional positive regard | Cooperative/trusting societies |
| **John Dewey** | Pragmatic reformer -- learning by doing, democratic values | Adaptive/pragmatic communities |
| **Dumbledore** | Wise leader -- strategic patience, moral complexity | Wisdom-driven leadership |
| **Yoda** | Ancient sage -- extreme patience, deep insight, low ego | Contemplative/meditative societies |
| **Ada Lovelace** | Visionary engineer -- mathematical creativity, future-seeing | Engineering/innovation societies |
| **Carl Sagan** | Science communicator -- wonder, accessibility, cosmic perspective | Knowledge-sharing communities |

Inject archetypes mid-simulation from Mission Control or via the API. The Ripple Tracker measures how outsider traits propagate through the population over generations.

---

## Running Experiments from Python

### Basic simulation

```python
from seldon.core.config import ExperimentConfig
from seldon.core.engine import SimulationEngine

config = ExperimentConfig(
    initial_population=100,
    generations_to_run=50,
    random_seed=42,
)

engine = SimulationEngine(config)
history = engine.run()  # list of GenerationMetrics

for gen in history:
    print(f"Gen {gen.generation}: pop={gen.population_size}, "
          f"breakthroughs={gen.breakthroughs}")
```

### A/B test

```python
from seldon.experiment.runner import ExperimentRunner

runner = ExperimentRunner()
comparison = runner.run_ab_test(
    config_a=ExperimentConfig(experiment_name="default"),
    config_b=ExperimentConfig(
        experiment_name="inverted",
        birth_order_rules={1: "best", 2: "weirdest", 3: "worst"},
    ),
)

for label, result in comparison.results.items():
    print(f"{label}: breakthroughs={result.total_breakthroughs}, "
          f"contribution={result.mean_contribution:.3f}")
```

### Parameter sweep

```python
runner = ExperimentRunner()
results = runner.run_parameter_sweep(
    ExperimentConfig(initial_population=50, generations_to_run=20),
    "trait_drift_rate",
    [0.001, 0.01, 0.05, 0.1, 0.2],
)

for label, result in results.items():
    print(f"drift={label}: final_pop={result.final_population_size}")
```

### Archetype injection

```python
config = ExperimentConfig(
    initial_population=50,
    generations_to_run=30,
    scheduled_injections=[
        {"generation": 5, "archetype": "einstein", "count": 3},
        {"generation": 10, "archetype": "curie", "count": 2},
    ],
)

engine = SimulationEngine(config)
history = engine.run()

report = engine.ripple_tracker.get_diffusion_report()
print(f"Outsider trait diffusion: {report['injections']} injections")
```

---

## API Reference

The REST API runs on port 8006 with 8 routers and 30+ endpoints.

### Simulation Management -- `/api/simulation`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/sessions` | Create a new simulation session |
| `GET` | `/sessions` | List all sessions |
| `GET` | `/sessions/{id}` | Get session details |
| `DELETE` | `/sessions/{id}` | Delete a session |
| `POST` | `/sessions/{id}/run` | Run to completion |
| `POST` | `/sessions/{id}/step` | Step N generations |
| `POST` | `/sessions/{id}/reset` | Reset to generation 0 |

### Agents -- `/api/agents`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/{session_id}` | List agents (filter by region, generation, birth order, alive/dead) |
| `GET` | `/{session_id}/{agent_id}` | Full agent detail (traits, history, memories, decisions) |
| `GET` | `/{session_id}/{agent_id}/family-tree` | Ancestry + descendant tree |

### Metrics -- `/api/metrics`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/{session_id}/generations` | All generation metrics (paginated) |
| `GET` | `/{session_id}/time-series/{field}` | Time series for a specific metric |
| `GET` | `/{session_id}/summary` | Summary statistics |

### Experiments -- `/api/experiments`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/presets` | List all presets with configs |
| `GET` | `/archetypes` | List all archetypes |
| `GET` | `/archetypes/{name}` | Full archetype detail with trait values |
| `POST` | `/compare` | Compare metrics across sessions |
| `POST` | `/inject-outsider` | Inject archetype or custom traits into a session |
| `GET` | `/{session_id}/ripple` | Outsider trait diffusion report |

### Settlements -- `/api/settlements`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/{session_id}/overview` | All settlements with composition |
| `GET` | `/{session_id}/viability/{location_id}` | Settlement viability assessment |
| `GET` | `/{session_id}/migration-history` | Migration event timeline |
| `GET` | `/{session_id}/settlement-composition/{location_id}` | Detailed settlement breakdown |

### Social Network -- `/api/network`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/{session_id}/graph` | Full social graph (nodes + edges + stats) |

### Advanced Analytics -- `/api/advanced`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/{session_id}/anomalies` | Z-score anomaly detection |
| `GET` | `/{session_id}/lore/overview` | Lore metrics + societal memory |
| `GET` | `/{session_id}/lore/meme-prevalence` | Cultural meme prevalence over time |
| `POST` | `/{session_id}/sensitivity` | Parameter sensitivity analysis |

### LLM -- `/api/llm`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/status` | Provider availability (Anthropic + Ollama) |
| `POST` | `/{session_id}/interview/{agent_id}` | In-character agent interview |
| `GET` | `/{session_id}/narrative/{generation}` | Prose narrative for a generation |
| `POST` | `/{session_id}/decision-explain` | Psychological decision analysis |

---

## Tests

381 tests covering all layers of the system.

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run a specific test file
pytest tests/test_inheritance.py

# Run with coverage
pytest --cov=seldon tests/
```

| Test File | Coverage |
|-----------|----------|
| `test_traits.py` | Trait system presets, indexing, vectors |
| `test_config.py` | Config serialization, from_dict, trait system init |
| `test_inheritance.py` | Birth order rules (worst, weirdest, best, random, twins) |
| `test_processing.py` | RSH Five Regions classification, R4/R5 distinction |
| `test_drift.py` | Trait drift, age effects, region effects |
| `test_attraction.py` | Attraction model components |
| `test_decision.py` | Utility-based decision model |
| `test_council.py` | Cognitive council modulation |
| `test_engine.py` | Full simulation engine, generation loop |
| `test_relationships.py` | Pairing, dissolution, infidelity, widowing |
| `test_fertility.py` | Birth spacing, mortality, societal pressure |
| `test_lore.py` | Memory transmission, fidelity decay, myths |
| `test_collector.py` | Metrics collection |
| `test_outsider.py` | Outsider injection, ripple tracking |
| `test_archetypes.py` | All 11 archetype vectors |
| `test_presets.py` | All 10 experiment presets |
| `test_runner.py` | A/B tests, parameter sweeps, multi-seed |
| `test_extensions.py` | Extension ABC, registry, all 6 extensions |
| `test_api.py` | REST API endpoints (sessions, agents, metrics) |
| `test_api_advanced.py` | Advanced endpoints (anomaly, lore, settlements, network, sensitivity) |
| `test_llm.py` | LLM client, prompts, interviewer, narrator (mocked) |

All LLM tests use mocked clients. No real API calls are made during testing.

---

## Project Structure

```
seldon-sandbox/
+-- src/seldon/
|   +-- core/           # TraitSystem, Agent, Config, Engine, Inheritance,
|   |                   #   Processing, Drift, Attraction, Decision, Council
|   +-- social/         # RelationshipManager, FertilityManager, LoreEngine
|   +-- metrics/        # MetricsCollector
|   +-- experiment/     # ExperimentRunner, Presets, Archetypes, OutsiderInterface
|   +-- extensions/     # SimulationExtension ABC, Registry, 6 extension modules
|   +-- llm/            # ClaudeClient, OllamaClient, Interviewer, Narrator, Prompts
|   +-- api/            # FastAPI app, SessionManager, Serializers, 8 routers
+-- frontend/           # React + TypeScript + Tailwind + Recharts + D3
|   +-- src/
|       +-- components/views/   # 12 dashboard views
|       +-- api/client.ts       # API client
|       +-- stores/             # Zustand state management
|       +-- types/              # TypeScript interfaces
+-- tests/              # 381 tests (pytest + hypothesis)
+-- examples/           # CLI example scripts
+-- docs/               # Architecture docs, conversation transcripts
+-- docker-compose.yml  # One-command deployment
+-- Dockerfile.backend  # Python 3.12 + FastAPI
+-- Dockerfile.frontend # Node 22 build + nginx serve
```

---

## Design Principles

1. **Everything is a slider.** No hardcoded assumptions. All thresholds, weights, rates, and rules come from `ExperimentConfig`. Every magic number is configurable.

2. **Track and visualize over time.** Every metric has a time series. Every agent records full trait, region, contribution, and suffering history every generation.

3. **Compare runs.** A/B testing and parameter sweeps are first-class. The system is designed for "what happens if I change X?" experiments.

4. **Core is simple, complexity is opt-in.** The base simulation runs with zero extensions. Geography, migration, resources, technology, culture, and conflict are all optional modules.

5. **Decisions are mathematical and explainable.** Every choice flows through a utility-based decision model: `U(a|P,x) = P^T * W_a * x + b_a` with softmax selection. Per-trait contribution analysis is recorded for every decision.

6. **Memory shapes behavior.** Generational lore with fidelity decay creates emergent mythology. Stories mutate as they pass between generations, eventually becoming myths that still influence behavior.

7. **LLM for interviews only.** The simulation runs on pure math. LLMs are a narrative layer for explaining what happened, not for determining what happens.

8. **Full determinism.** Every random operation flows through a seeded `numpy.random.Generator`. Same seed = same results. Always.

---

## Key Experiments the System Supports

- **Birth order hypothesis**: Does 1st=worst, 2nd=weirdest, 3rd=best produce emergent structures? What about inverted rules?
- **Optimal processing mix**: What R1-R5 proportion maximizes societal contribution?
- **Settlement composition**: What personality mix makes a new settlement succeed or fail?
- **Outsider disruption**: How quickly do foreign traits propagate? Do they improve or destabilize?
- **Cultural meme effects**: Does a "tortured genius" meme maintain the sacrificial population?
- **Lore degradation**: How do distorted memories affect collective behavior?
- **Archetype societies**: What emerges from a society of Einsteins? Of Curies + Fred Rogers?
- **Resource scarcity pressure**: How does scarcity change processing region distributions?

---

## License

MIT License. See [LICENSE](LICENSE) for details.

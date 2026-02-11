# The Seldon Sandbox

**A multi-generational societal simulation engine for exploring how personality traits, cognitive processing styles, genetic inheritance, social hierarchies, and environmental pressures produce emergent social structures across generations.**

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
- [Genetics & Epigenetics](#genetics--epigenetics)
- [Social Hierarchies & Mentorship](#social-hierarchies--mentorship)
- [Trait System](#trait-system)
- [Extensions](#extensions)
- [Outsider Injection & Tracking](#outsider-injection--tracking)
- [LLM Integration](#llm-integration)
- [Experiment Presets](#experiment-presets)
- [Archetypes](#archetypes)
- [Running Experiments from Python](#running-experiments-from-python)
- [API Reference](#api-reference)
- [Session Persistence](#session-persistence)
- [Tests](#tests)
- [Project Structure](#project-structure)
- [Design Principles](#design-principles)

---

## Core Ideas

The simulation models a population of agents, each defined by an N-dimensional personality trait vector (15 traits in compact mode, 50 in full mode). Every generation, agents:

1. **Age and drift** -- traits shift over time based on experience and processing region effects
2. **Update epigenetics** -- environmental markers activate or deactivate based on conditions (when genetics enabled)
3. **Get classified** into one of five cognitive processing regions (from the RSH model)
4. **Contribute** to society -- output depends on their processing region, creativity, and resilience
5. **Form and dissolve** relationships based on attraction (similarity, complementarity, chemistry)
6. **Reproduce** -- children inherit traits via birth-order rules (1st=worst, 2nd=weirdest, 3rd=best), optionally with allele-based genetic inheritance
7. **Transmit memories and lore** -- stories degrade over generations, creating emergent myths
8. **Participate in social dynamics** -- hierarchies form, mentorships develop, roles are assigned
9. **Die** -- from age, burnout, conflict, or environmental pressure

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
| `backend` | 8006 | FastAPI simulation engine + REST API (Python source volume-mounted for live changes) |
| `frontend` | 3006 | React dashboard (nginx serves statically built assets, proxies `/api` to backend) |

### Important: Frontend vs Backend changes in Docker

- **Backend changes** are picked up automatically -- the `src/` directory is volume-mounted into the container.
- **Frontend changes require a rebuild** -- the frontend Docker image compiles TypeScript and bundles assets at build time. After modifying frontend code:

```bash
docker compose build frontend --no-cache
docker compose up -d
```

### Session Persistence in Docker

Session data is stored in a SQLite database on a named Docker volume (`seldon-data`), so **sessions survive container restarts and rebuilds**. The database is stored at `/app/data/seldon.db` inside the container.

```bash
# Sessions persist across restarts
docker compose restart backend

# Sessions persist across rebuilds
docker compose up --build

# To wipe session data, remove the volume
docker compose down -v
```

### Environment variables

Create a `.env` file in the project root (already gitignored):

```env
# Optional: enables Anthropic Claude for agent interviews/narratives
ANTHROPIC_API_KEY=sk-ant-...

# Optional: custom Ollama host (default auto-detects Docker vs local)
OLLAMA_HOST=http://host.docker.internal:11434

# Optional: custom database path (default: data/seldon.db)
SELDON_DB_PATH=data/seldon.db
```

The API key is passed into the Docker container automatically via the `env_file` directive in `docker-compose.yml`. Without it, the simulation runs normally -- LLM features just show an "unavailable" banner.

### Ollama (local LLM)

If you prefer running LLM features without an API key, install [Ollama](https://ollama.com) on your host machine and pull a model:

```bash
ollama pull llama3.2
```

The Docker container automatically connects to Ollama on the host via `host.docker.internal` (configured with `extra_hosts: host.docker.internal:host-gateway` in docker-compose). You can also set `OLLAMA_HOST` to point to a custom Ollama instance. In the dashboard, go to the Interview view's Settings tab and select "Ollama" as the provider.

### Docker Compose configuration

```yaml
services:
  backend:
    build:
      context: .
      dockerfile: Dockerfile.backend
    ports:
      - "8006:8006"
    volumes:
      - ./src:/app/src              # Live backend code reload
      - ./tests:/app/tests          # Live test reload
      - ./.env:/app/.env:ro         # Environment file
      - seldon-data:/app/data       # Persistent session storage
    env_file:
      - .env                        # Loads ANTHROPIC_API_KEY, OLLAMA_HOST, etc.
    extra_hosts:
      - "host.docker.internal:host-gateway"  # Required for Ollama access

  frontend:
    build:
      context: .
      dockerfile: Dockerfile.frontend
    ports:
      - "3006:3006"
    depends_on:
      - backend

volumes:
  seldon-data:                      # Named volume for SQLite persistence
```

### Rebuilding

```bash
# Rebuild everything
docker compose up --build

# Rebuild only frontend (after JS/TS changes)
docker compose build frontend --no-cache && docker compose up -d

# Restart only backend (after Python changes)
docker compose restart backend

# View logs
docker compose logs -f backend
docker compose logs -f frontend
```

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

# Start the API server (with hot reload)
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

598 tests covering the core engine, genetics, social dynamics, extensions, API endpoints, persistence, and LLM integration. All LLM tests use mocked clients -- no real API calls.

---

## The Dashboard

The web dashboard at [http://localhost:3006](http://localhost:3006) provides 18 interactive views organized into 6 sections:

### Core Views

| View | What It Shows |
|------|--------------|
| **Mission Control** | Create sessions, configure all parameters (population, inheritance, regions, extensions, genetics, social dynamics, diplomacy, economics, environment), build and inject custom outsiders, step through or run generations. The command center for everything. |
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
| **Outsider Tracker** | Registry of all injected outsiders, ripple impact charts (outsider fraction over time), selected outsider detail with descendants. |

### Social Views

| View | What It Shows |
|------|--------------|
| **Hierarchy** | Social status distribution, role breakdown (leader, innovator, mediator, worker, outsider_bridge), top agents by influence score, mentorship chains. |
| **Communities** | Community personality radar charts, cohesion scores, diplomatic relations table with alliance/rivalry/neutral standings. |

### Economy Views

| View | What It Shows |
|------|--------------|
| **Economics** | GDP per community, occupation breakdown (pie chart), wealth distribution by percentile, trade routes with volumes. |
| **Environment** | Current season indicator, climate state per location (temperature/rainfall), event timeline with severity, active disease tracker. |

### Science Views

| View | What It Shows |
|------|--------------|
| **Genetics** | Allele frequency stacked bar chart per locus, epigenetic marker prevalence bars, trait-gene correlation horizontal bars. Gracefully degrades when genetics not enabled. |

### LLM Views

| View | What It Shows |
|------|--------------|
| **Agent Interview** | Four-tab interface: (1) Chat with any agent in-character, (2) Generate prose narratives for any generation, (3) Get psychological explanations of agent decisions, (4) Configure LLM provider (Anthropic or Ollama), model selection, API key management, test connection. |

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
    |   +-- Agent ................. dataclass with traits, history, lore, decisions, genome
    |   +-- DecisionModel ......... utility-based: U(a|P,x) = P^T * W_a * x + b_a
    |   +-- CognitiveCouncil ...... optional 8-voice processing modulation
    |   +-- InheritanceEngine ..... birth order rules + optional genetic inheritance
    |   +-- ProcessingClassifier .. RSH Five Regions assignment
    |   +-- TraitDriftEngine ...... experience + age-based trait changes
    |   +-- AttractionModel ....... similarity, complementarity, chemistry
    |   +-- SimulationEngine ...... 9-phase generation loop with extension hooks
    |   +-- GeneticModel .......... allele pairs, crossover, mutation, expression
    |   +-- EpigeneticModel ....... environmental markers, transgenerational inheritance
    |   +-- GeneticAttribution .... lineage tracking, trait-gene correlation
    |
    +-- SOCIAL
    |   +-- RelationshipManager ... pairing, dissolution, infidelity, widowing
    |   +-- FertilityManager ...... birth spacing, maternal/child mortality, pressure
    |   +-- LoreEngine ............ memory transmission, fidelity decay, myth formation
    |   +-- SocialHierarchyManager  status, influence, role assignment
    |   +-- MentorshipManager ..... matching, skill transfer, dissolution, chains
    |
    +-- EXTENSIONS (optional, via ExtensionRegistry)
    |   +-- Geography ............. hexagonal grid, settlements, terrain
    |   +-- Migration ............. settlement viability, group migration (requires Geography)
    |   +-- Resources ............. resource production, distribution, scarcity
    |   +-- Technology ............ tech advancement, tool access
    |   +-- Culture ............... cultural memes, transmission, dominance
    |   +-- Conflict .............. personality-based triggers, trait-influenced resolution
    |   +-- Social Dynamics ....... hierarchy + mentorship wired into engine hooks
    |   +-- Diplomacy ............. alliances, rivalries, cultural exchange (requires Geography)
    |   +-- Economics ............. production, trade, wealth distribution, occupations
    |   +-- Environment ........... seasons, climate, drought, plague, disease
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
        +-- FastAPI REST API ...... 13 routers, 50+ endpoints
        +-- SessionManager ........ in-memory sessions with SQLite persistence
        +-- SessionStore .......... SQLite CRUD, zlib-compressed state blobs
        +-- React Dashboard ....... 18 views, real-time updates
```

### Generation Loop (9 phases per generation)

1. **Age & Trait Drift** -- Agents age, traits shift based on drift rate and processing region effects
2. **Epigenetic Updates** -- Environmental markers activate/deactivate based on agent conditions (Phase 1.5, when genetics enabled)
3. **Processing Region Updates** -- Reclassify into R1-R5; update dominant cognitive voice
4. **Contribution & Breakthroughs** -- Calculate output, detect breakthroughs, create memories
5. **Relationship Dynamics** -- Process dissolutions, form new pairs via the decision model
6. **Reproduction** -- Paired agents produce children; traits inherited via birth order rules (optionally with allele-based genetics); lore transmitted
7. **Lore Evolution** -- Societal-level memory consensus, conflict, mutation, myth formation
8. **Mortality** -- Age/burnout/extension-modified death checks; handle widowing
9. **Extension Hooks** -- Extensions fire at 8 lifecycle points: simulation start, generation start/end, agent created, modify attraction, modify mortality, modify decision utilities, and collect metrics

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

## Genetics & Epigenetics

An optional layer that adds allele-based genetic inheritance and environmental epigenetic markers to the simulation. Enable from Mission Control's "Genetics & Epigenetics" config section.

### Genetic Model

When genetics is enabled, each agent carries a genome of 10 gene loci (allele pairs):

| Locus | Mapped Trait | Effect |
|-------|-------------|--------|
| CREA_1 | creativity | Dominant allele boosts creative output |
| RESI_1 | resilience | Affects recovery and burnout resistance |
| NEUR_1 | neuroticism | Influences emotional stability |
| EXTR_1 | extraversion | Social engagement tendency |
| CONS_1 | conscientiousness | Work ethic and reliability |
| OPEN_1 | openness | Curiosity and intellectual exploration |
| DEPT_1 | depth_drive | Processing depth tendency |
| AGRE_1 | agreeableness | Cooperation and social harmony |
| AMBI_1 | ambition | Drive and goal orientation |
| EMPA_1 | empathy | Emotional understanding |

Allele expression follows Mendelian genetics: AA = +1.0 (fully dominant), Aa = +0.5 (heterozygous), aa = -1.0 (recessive). During reproduction, genomes undergo crossover and mutation (rates configurable via `genetics_config`).

### Epigenetic Model

Five environmental markers can activate or deactivate across generations based on agent experiences:

- **stress_resilience** -- activates under sustained suffering
- **creative_amplification** -- activates in high-creativity environments
- **social_withdrawal** -- activates from social isolation
- **resource_conservation** -- activates during resource scarcity
- **trauma_sensitivity** -- activates from traumatic events

Epigenetic markers exhibit transgenerational inheritance -- a marker activated in a parent can be passed to children at a configurable rate. This creates emergent multi-generational adaptation patterns.

### Configuration

```python
config = ExperimentConfig(
    genetics_config={
        "genetics_enabled": True,
        "mutation_rate": 0.01,
        "crossover_rate": 0.5,
        "gene_trait_influence": 0.3,
    },
    epigenetics_config={
        "epigenetics_enabled": True,
        "transgenerational_rate": 0.3,
        "activation_threshold_generations": 2,
        "max_active_markers": 3,
    },
)
```

When genetics is disabled, the simulation falls back to standard birth-order inheritance with zero behavioral change.

---

## Social Hierarchies & Mentorship

An optional extension that adds social status, influence scoring, role assignment, and mentorship systems.

### Hierarchy

Every agent receives a **social status** score (0-1) based on their contributions, age, and social connections. Status determines role assignment:

| Role | Description |
|------|-------------|
| **leader** | High status, high influence -- shapes community direction |
| **innovator** | High creativity, drives breakthroughs |
| **mediator** | High agreeableness + empathy, resolves conflicts |
| **worker** | Steady contributors, backbone of the community |
| **outsider_bridge** | Injected outsiders who connect communities |
| **unassigned** | No clear role yet (typically young agents) |

An **influence score** decays over time (configurable rate) and determines how much an agent's decisions affect others.

### Mentorship

Experienced agents can mentor younger ones:

- **Matching**: Based on skill complementarity and social proximity
- **Skill Transfer**: Mentees gain trait boosts from mentors over time
- **Dissolution**: Mentorships end when the mentee outgrows the mentor or either dies
- **Chains**: A mentor's mentor's influence can trace through multi-generation lineages

Enable via the "Social Dynamics" extension toggle on Mission Control.

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
    extensions_enabled=["geography", "migration", "resources", "social_dynamics", "economics"],
    extensions={
        "geography": {"grid_size": 10},
        "resources": {"base_production": 1.0},
        "economics": {"trade_distance_cost": 0.1},
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
| **Conflict** | Personality-based conflict triggers (dominance clashes, trust betrayals), trait-influenced resolution (submission, compromise, separation, escalation) | None |
| **Social Dynamics** | Social hierarchy (status, influence, roles), mentorship (matching, skill transfer, chains) | None |
| **Diplomacy** | Alliance formation, rivalry detection, cultural exchange between settlements | Geography |
| **Economics** | Production, trade routes, wealth distribution, occupations, poverty effects | None |
| **Environment** | Seasonal cycles, climate drift, droughts, plagues, disease tracking | Geography |

Extensions hook into the generation loop at 8 points (simulation start, generation start/end, agent created, modify attraction/mortality/decisions, collect metrics). They can modify how agents make decisions, who they're attracted to, and how likely they are to die -- but the core simulation math stays the same.

### Configuring Extensions from Mission Control

The dashboard provides collapsible config panels for each extension group:

- **Genetics & Epigenetics** -- mutation rate, gene-trait influence, transgenerational rate
- **Social Dynamics** -- status weights, influence decay, mentorship toggle, max mentees
- **Diplomacy** -- alliance/rivalry thresholds, cultural exchange rate
- **Economics** -- base production, trade distance cost, poverty threshold
- **Environment** -- season toggle/length, drought/plague probability, climate drift rate

Dependencies are enforced automatically: enabling Migration auto-enables Geography; enabling Diplomacy or Environment auto-enables Geography.

---

## Outsider Injection & Tracking

Outsiders are agents with pre-defined trait profiles injected into an existing population mid-simulation. They test how foreign personality types propagate through and influence a community.

### Outsider Builder (Mission Control)

The dashboard provides a two-mode outsider builder:

**Archetype Mode**: Select from 11 pre-defined personality archetypes (Da Vinci, Einstein, etc.), optionally override the name, set an injection generation, and adjust noise sigma.

**Custom Build Mode**: Build an outsider from scratch:
- Set name, gender, and age
- Adjust each trait individually via per-trait sliders (trait names fetched dynamically from the backend)
- Choose the injection generation (default: current generation)
- Preview the agent before injection

### Outsider Tracker View

A dedicated view for monitoring outsider impact after injection:

1. **Outsider Registry** -- list of all injected outsiders with status, processing region, generation, and age
2. **Ripple Impact** -- outsider fraction over time (line chart), annotated with injection points
3. **Selected Outsider Detail** -- origin, injection generation, trait distance from population mean, descendant count, descendant list

### API

```python
# Inject by archetype
POST /api/experiments/inject-outsider
{
    "session_id": "abc123",
    "archetype": "einstein",
    "name": "Albert",          # optional name override
    "injection_generation": 5   # optional, default = current generation
}

# Inject custom traits
POST /api/experiments/inject-outsider
{
    "session_id": "abc123",
    "custom_traits": {"creativity": 0.95, "resilience": 0.8, "depth_drive": 0.9},
    "name": "Custom Agent",
    "gender": "female",
    "age": 30
}

# Track impact
GET /api/experiments/{session_id}/outsiders           # List all outsiders
GET /api/experiments/{session_id}/outsiders/{id}/impact  # Outsider impact detail
GET /api/experiments/{session_id}/ripple              # Trait diffusion report
```

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
| **Anthropic (Claude)** | Set `ANTHROPIC_API_KEY` env var or enter in Settings tab | Highest quality narratives |
| **Ollama (Local)** | Install Ollama, pull a model (e.g., `ollama pull llama3.2`) | Free, private, no API key needed |

Switch providers in the Interview view's Settings tab. The settings panel provides:
- Card-based provider selection (Anthropic/Ollama)
- Model dropdown for both providers (Ollama auto-detects installed models)
- API key management for Anthropic (enter/delete at runtime)
- Configurable Ollama base URL
- Test Connection button to verify setup

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

Inject archetypes mid-simulation from Mission Control's Outsider Builder or via the API. The Ripple Tracker and Outsider Tracker views measure how outsider traits propagate through the population over generations.

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

### With genetics and extensions

```python
config = ExperimentConfig(
    initial_population=80,
    generations_to_run=30,
    random_seed=42,
    extensions_enabled=["geography", "migration", "resources", "social_dynamics", "economics"],
    genetics_config={"genetics_enabled": True, "mutation_rate": 0.01},
    epigenetics_config={"epigenetics_enabled": True},
    extensions={
        "geography": {"grid_size": 8},
        "economics": {"base_production_rate": 1.5},
    },
)

engine = SimulationEngine(config)
history = engine.run()
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

The REST API runs on port 8006 with 13 routers and 50+ endpoints.

### Simulation Management -- `/api/simulation`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/sessions` | Create a new simulation session (accepts full config) |
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
| `GET` | `/trait-names` | Ordered list of trait names from the trait system |
| `POST` | `/compare` | Compare metrics across sessions |
| `POST` | `/inject-outsider` | Inject archetype or custom outsider (with name, gender, age, injection gen) |
| `GET` | `/{session_id}/ripple` | Outsider trait diffusion report |
| `GET` | `/{session_id}/outsiders` | List all injected outsiders in a session |
| `GET` | `/{session_id}/outsiders/{agent_id}/impact` | Outsider impact detail (descendants, trait distance) |

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

### Social -- `/api/social`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/{session_id}/hierarchy` | Social status distribution and stats |
| `GET` | `/{session_id}/roles` | Role breakdown across the population |
| `GET` | `/{session_id}/mentorship` | Active mentorship pairs and chains |
| `GET` | `/{session_id}/influence-map` | Agent influence scores and rankings |

### Communities -- `/api/communities`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/{session_id}/communities` | Community profiles with personality and cohesion |
| `GET` | `/{session_id}/diplomacy` | Diplomatic relations between communities |

### Economics -- `/api/economics`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/{session_id}/overview` | Economic overview (GDP, Gini, population stats) |
| `GET` | `/{session_id}/trade-routes` | Active trade routes with volumes |
| `GET` | `/{session_id}/markets` | Market state per settlement |
| `GET` | `/{session_id}/wealth-distribution` | Wealth percentile breakdown |
| `GET` | `/{session_id}/occupations` | Occupation type distribution |

### Environment -- `/api/environment`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/{session_id}/climate` | Climate state (temperature, rainfall, season) |
| `GET` | `/{session_id}/events` | Environmental event history (droughts, plagues) |
| `GET` | `/{session_id}/disease` | Active disease tracking |

### Genetics -- `/api/genetics`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/{session_id}/allele-frequencies` | Per-locus allele frequency breakdown |
| `GET` | `/{session_id}/epigenetic-prevalence` | Epigenetic marker activation rates |
| `GET` | `/{session_id}/trait-gene-correlation` | Pearson correlation between gene expression and trait values |

### LLM -- `/api/llm`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/status` | Provider availability (Anthropic + Ollama) |
| `POST` | `/api-key` | Set Anthropic API key at runtime |
| `DELETE` | `/api-key` | Clear runtime API key |
| `POST` | `/ollama-url` | Set custom Ollama base URL |
| `POST` | `/test-connection` | Test LLM provider connectivity |
| `POST` | `/{session_id}/interview/{agent_id}` | In-character agent interview |
| `GET` | `/{session_id}/narrative/{generation}` | Prose narrative for a generation |
| `POST` | `/{session_id}/decision-explain` | Psychological decision analysis |

---

## Session Persistence

Sessions are automatically persisted to a SQLite database, so they survive backend restarts, Docker rebuilds, and container recreation.

### How It Works

- **Auto-save**: Every mutation (create, step, run, reset, delete) saves the session to SQLite
- **Lazy loading**: On startup, only session metadata is loaded; full state is deserialized on first access
- **Compressed storage**: Full session state (all agents, metrics history, engine state) is serialized as zlib-compressed JSON blobs
- **Graceful fallback**: If the database can't be created, the system works in memory-only mode with no errors

### What's Persisted

- All agent data (traits, history, relationships, genetics, social status, decisions, memories)
- Full metrics history (all GenerationMetrics for every generation)
- Session metadata (name, status, generation, population size)
- Engine state (next agent ID, previous region tracking for transitions)
- Experiment configuration

### What's NOT Persisted (Rebuilt on Load)

- Extension internal state (settlements, resource pools, trade markets) -- rebuilt via `on_simulation_start()`
- Lore engine's societal consensus list -- rebuilt from agent memories
- RippleTracker snapshots -- outsiders still tracked via agent `is_outsider` flag
- Exact RNG state -- reseeded deterministically as `seed + current_generation`

### Configuration

```bash
# Environment variable (default: data/seldon.db)
export SELDON_DB_PATH=data/seldon.db

# Docker: uses named volume (seldon-data) at /app/data
# Local dev: creates data/ directory in project root

# Disable persistence (in-memory only)
export SELDON_DB_PATH=""
```

### Database Schema

A single `sessions` table with metadata columns for fast listing and a zlib-compressed blob for full state:

```sql
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'created',
    current_generation INTEGER NOT NULL DEFAULT 0,
    max_generations INTEGER NOT NULL DEFAULT 0,
    population_size INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    config_json TEXT NOT NULL,
    state_blob BLOB
);
```

---

## Tests

598 tests across 28 test files covering all layers of the system.

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
| `test_extensions.py` | Extension ABC, registry, all extension modules |
| `test_api.py` | REST API endpoints (sessions, agents, metrics) |
| `test_api_advanced.py` | Advanced endpoints (anomaly, lore, settlements, network, sensitivity) |
| `test_llm.py` | LLM client, prompts, interviewer, narrator (mocked) |
| `test_social_dynamics.py` | Hierarchy, mentorship, influence, role assignment |
| `test_genetics.py` | Genetic model, alleles, crossover, mutation, expression |
| `test_genetics_api.py` | Genetics router, outsider endpoints, custom injection, gender |
| `test_community.py` | Community detection, diplomatic relations |
| `test_economics.py` | Production, trade, wealth distribution |
| `test_environment.py` | Seasons, climate, events, disease |
| `test_persistence.py` | Agent serialization roundtrips, SessionStore CRUD, state blob compression, full restart integration |
| `conftest.py` | Shared fixtures (DB isolation via temp path) |

All LLM tests use mocked clients. No real API calls are made during testing.

---

## Project Structure

```
seldon-sandbox/
+-- src/seldon/
|   +-- core/           # TraitSystem, Agent, Config, Engine, Inheritance,
|   |                   #   Processing, Drift, Attraction, Decision, Council,
|   |                   #   GeneticModel, EpigeneticModel, GeneticAttribution
|   +-- social/         # RelationshipManager, FertilityManager, LoreEngine,
|   |                   #   SocialHierarchyManager, MentorshipManager
|   +-- metrics/        # MetricsCollector
|   +-- experiment/     # ExperimentRunner, Presets, Archetypes, OutsiderInterface
|   +-- extensions/     # SimulationExtension ABC, Registry, 10 extension modules
|   |                   #   (geography, migration, resources, technology, culture,
|   |                   #    conflict, social_dynamics, diplomacy, economics, environment)
|   +-- llm/            # ClaudeClient, OllamaClient, Interviewer, Narrator, Prompts
|   +-- api/            # FastAPI app, SessionManager, SessionStore (SQLite persistence),
|                       #   Serializers, 13 routers (simulation, agents, metrics,
|                       #   experiments, settlements, network, advanced, llm, social,
|                       #   communities, economics, environment, genetics)
+-- frontend/           # React + TypeScript + Tailwind v4 + Recharts + D3
|   +-- src/
|       +-- components/
|       |   +-- views/         # 18 dashboard views
|       |   +-- layout/        # Sidebar, MainLayout
|       |   +-- shared/        # Reusable components
|       +-- api/client.ts      # API client (50+ functions)
|       +-- stores/            # Zustand state management
|       +-- types/             # TypeScript interfaces
+-- tests/              # 598 tests (pytest)
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

4. **Core is simple, complexity is opt-in.** The base simulation runs with zero extensions. All 10 environmental, social, and economic extensions are optional modules. Genetics and epigenetics are opt-in layers that fall back gracefully.

5. **Decisions are mathematical and explainable.** Every choice flows through a utility-based decision model: `U(a|P,x) = P^T * W_a * x + b_a` with softmax selection. Per-trait contribution analysis is recorded for every decision.

6. **Memory shapes behavior.** Generational lore with fidelity decay creates emergent mythology. Stories mutate as they pass between generations, eventually becoming myths that still influence behavior.

7. **LLM for interviews only.** The simulation runs on pure math. LLMs are a narrative layer for explaining what happened, not for determining what happens.

8. **Full determinism.** Every random operation flows through a seeded `numpy.random.Generator`. Same seed = same results. Always.

9. **Sessions are durable.** Every session mutation auto-saves to SQLite. Restarts don't lose work. Lazy loading keeps startup fast. Persistence failures never crash the system.

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
- **Genetic drift vs selection**: Do allele frequencies shift toward optimal or random distributions?
- **Epigenetic adaptation**: Do trauma markers persist and help future generations cope?
- **Social hierarchy emergence**: What personality mixes produce stable vs. chaotic hierarchies?
- **Economic inequality**: How does trade network topology affect wealth distribution?

---

## License

MIT License. See [LICENSE](LICENSE) for details.

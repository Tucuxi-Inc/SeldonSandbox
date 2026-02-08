# The Seldon Sandbox: Claude Code Handoff Document

## Project Overview

The Seldon Sandbox is a **multi-generational societal simulation engine** designed as an experimentation platform. It models how personality traits, cognitive processing styles, and environmental pressures interact to produce emergent social structures across generations.

**Key Design Principles:**
1. **Everything is a slider** — No hardcoded assumptions; all parameters are tunable
2. **Track and visualize over time** — Every metric has a time series
3. **Compare runs** — A/B testing of parameter configurations
4. **Core is simple, complexity is opt-in** — Extension architecture for future features

---

## Source Documents

The following architecture document contains the complete specification:

- **`seldon_sandbox_v2.3_architecture.md`** — Master architecture with:
  - 15-trait personality system
  - RSH Five Regions cognitive processing model
  - Configurable birth order inheritance rules
  - Trait drift mechanics
  - Attraction/pairing model
  - Extension architecture for future modules (geography, resources, migration, etc.)

Read this document thoroughly before starting implementation.

---

## Project Structure

```
seldon-sandbox/
├── README.md
├── pyproject.toml              # Use Poetry or pip
├── requirements.txt
│
├── src/
│   └── seldon/
│       ├── __init__.py
│       │
│       ├── core/               # Core simulation components
│       │   ├── __init__.py
│       │   ├── config.py       # ExperimentConfig dataclass
│       │   ├── traits.py       # TraitSystem (15 traits)
│       │   ├── agent.py        # Agent dataclass
│       │   ├── inheritance.py  # InheritanceEngine (birth order rules)
│       │   ├── processing.py   # ProcessingClassifier (RSH Five Regions)
│       │   ├── drift.py        # TraitDriftEngine
│       │   ├── attraction.py   # AttractionModel
│       │   └── engine.py       # SimulationEngine (main loop)
│       │
│       ├── metrics/            # Tracking and analysis
│       │   ├── __init__.py
│       │   ├── collector.py    # MetricsCollector
│       │   └── export.py       # Export for visualization
│       │
│       ├── experiment/         # Experiment running
│       │   ├── __init__.py
│       │   ├── runner.py       # ExperimentRunner
│       │   └── presets.py      # Preset configurations
│       │
│       ├── extensions/         # Optional modules (build later)
│       │   ├── __init__.py
│       │   ├── base.py         # SimulationExtension ABC
│       │   ├── registry.py     # ExtensionRegistry
│       │   ├── geography.py    # (stub for now)
│       │   ├── resources.py    # (stub for now)
│       │   ├── migration.py    # (stub for now)
│       │   └── culture.py      # (stub for now)
│       │
│       └── llm/                # LLM integration (build later)
│           ├── __init__.py
│           └── interviewer.py  # Agent interviews
│
├── tests/
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_traits.py
│   ├── test_inheritance.py
│   ├── test_processing.py
│   ├── test_drift.py
│   ├── test_attraction.py
│   ├── test_engine.py
│   └── test_metrics.py
│
├── notebooks/                  # Jupyter notebooks for exploration
│   ├── 01_basic_simulation.ipynb
│   ├── 02_birth_order_experiments.ipynb
│   └── 03_visualization.ipynb
│
├── examples/
│   ├── run_baseline.py
│   ├── run_ab_test.py
│   └── parameter_sweep.py
│
└── docs/
    └── architecture_v2.3.md    # Copy of architecture doc
```

---

## Implementation Order

### Phase 1: Foundation (Priority: P0)

Build these first, in order:

#### 1. `src/seldon/core/traits.py`
```python
# The 15-trait system
# - TraitSystem class with TRAITS dict and index constants
# - Helper methods: trait_name(index), trait_index(name)
```

#### 2. `src/seldon/core/config.py`
```python
# ExperimentConfig dataclass
# - All tunable parameters with defaults
# - Nested dicts for: birth_order_rules, region_thresholds, 
#   trait_stability, region_effects, attraction_weights, etc.
# - to_dict() and from_dict() methods for serialization
# - Extension support: extensions_enabled, extensions dict
```

#### 3. `src/seldon/core/agent.py`
```python
# Agent dataclass
# - Core fields: id, name, age, generation, birth_order, traits, etc.
# - Processing region enum and current region
# - State: suffering, burnout_level, is_alive
# - Relationships: partner_id, parent_ids, children_ids
# - History lists: trait_history, region_history, contribution_history
# - Extension hooks: location_id, resource_holdings, extension_data dict
```

#### 4. `src/seldon/core/processing.py`
```python
# ProcessingRegion enum (5 regions from RSH)
# ProcessingClassifier class
# - classify(traits) -> ProcessingRegion
# - process_region_transitions(agent) -> ProcessingRegion
# - Uses thresholds from config (not hardcoded!)
```

#### 5. `src/seldon/core/inheritance.py`
```python
# InheritanceEngine class
# - inherit(parent1, parent2, birth_order, population) -> np.ndarray
# - Configurable rules: "worst", "weirdest", "best", "random_weighted", "average"
# - Environmental noise and mutation application
# - DESIRABILITY dict for worst/best calculations
```

#### 6. `src/seldon/core/drift.py`
```python
# TraitDriftEngine class
# - drift_traits(agent, experiences) -> np.ndarray
# - apply_region_effects(agent) -> np.ndarray
# - Age-based stability
# - Trait-specific stability from config
```

#### 7. `src/seldon/core/attraction.py`
```python
# AttractionModel class
# - calculate(agent1, agent2) -> float
# - Components: similarity, complementarity, universal_attractiveness,
#   social_proximity, age_compatibility, random_chemistry
# - All weights from config
```

#### 8. `src/seldon/core/engine.py`
```python
# SimulationEngine class
# - __init__(config, extensions_registry)
# - run(generations) -> MetricsCollector
# - _run_generation(gen, population) -> events dict
# - Extension hooks at each phase
# - Phases: age/drift, regions, contribution, pairing, reproduction, mortality
```

### Phase 2: Metrics & Running (Priority: P1)

#### 9. `src/seldon/metrics/collector.py`
```python
# GenerationMetrics dataclass (all per-generation stats)
# MetricsCollector class
# - collect(generation, population, events) -> GenerationMetrics
# - get_time_series(metric_name) -> List[float]
# - export_for_visualization() -> dict
```

#### 10. `src/seldon/experiment/runner.py`
```python
# ExperimentRunner class
# - run_experiment(config) -> MetricsCollector
# - compare_experiments(names, metric) -> dict
# - run_ab_test(config_a, config_b, n_runs) -> statistical comparison
```

#### 11. `src/seldon/experiment/presets.py`
```python
# Preset configuration functions:
# - config_baseline()
# - config_no_birth_order()
# - config_inverted_birth_order()
# - config_high_sacrificial()
# - config_no_recovery()
# - config_high_trait_drift()
# - config_opposites_attract_only()
```

### Phase 3: Extensions Framework (Priority: P2)

#### 12. `src/seldon/extensions/base.py`
```python
# SimulationExtension ABC
# - All hook methods with default pass implementations
```

#### 13. `src/seldon/extensions/registry.py`
```python
# ExtensionRegistry class
# - register(), enable(), disable(), get_enabled()
```

#### 14. Extension stubs (just the class skeleton + get_default_config)
- geography.py
- resources.py
- migration.py
- culture.py

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

### 2. Traits Are 15-Dimensional NumPy Arrays

```python
# Traits are always np.ndarray of shape (15,)
# Values are floats in [0, 1]
agent.traits = np.array([0.5, 0.7, 0.3, ...])  # 15 values

# Use TraitSystem constants for indexing
from seldon.core.traits import TraitSystem
conscientiousness = agent.traits[TraitSystem.CONSCIENTIOUSNESS]
```

### 3. Processing Regions Are Enums

```python
from enum import Enum

class ProcessingRegion(Enum):
    UNDER_PROCESSING = "under_processing"
    OPTIMAL = "optimal"
    DEEP = "deep"
    SACRIFICIAL = "sacrificial"
    PATHOLOGICAL = "pathological"
```

### 4. History Tracking Is Essential

Every agent needs history lists for visualization:

```python
@dataclass
class Agent:
    trait_history: List[np.ndarray] = field(default_factory=list)
    region_history: List[ProcessingRegion] = field(default_factory=list)
    contribution_history: List[float] = field(default_factory=list)
    suffering_history: List[float] = field(default_factory=list)
```

Append to these each generation.

### 5. Extension Hooks Pattern

The simulation engine calls extension hooks at defined points:

```python
def _run_generation(self, gen, population):
    # Hook: generation start
    for ext in self.extensions.get_enabled():
        ext.on_generation_start(gen, population, self.config)
    
    # ... core logic ...
    
    # Hook: modify mortality
    for ext in self.extensions.get_enabled():
        mortality_rate = ext.modify_mortality(agent, mortality_rate, self.config)
    
    # Hook: generation end
    for ext in self.extensions.get_enabled():
        ext.on_generation_end(gen, population, self.config)
```

### 6. Birth Order Rules

The inheritance engine supports these rule types:
- `"worst"` — Child gets parent's worse trait value (based on desirability)
- `"best"` — Child gets parent's better trait value
- `"weirdest"` — Child gets whichever parent is farther from population mean
- `"random_weighted"` — Random mix of both parents
- `"average"` — Simple average of both parents

Rules are configurable per birth order position.

### 7. RSH Five Regions Summary

From Kevin's Reasoning Saturation Hypothesis paper:

| Region | Description | Key Characteristic |
|--------|-------------|-------------------|
| Under-Processing | Quick, shallow decisions | Low depth, low contribution |
| Optimal | Balanced, efficient | Peak efficiency, sustainable |
| Deep | Thorough, costly | High contribution, recoverable |
| Sacrificial | Obsessive, productive suffering | High contribution despite high cost (Van Gogh, Curie) |
| Pathological | Obsessive, unproductive suffering | High cost, no output (OCD loops, rumination) |

The R4 vs R5 distinction is critical: both suffer, but R4 produces breakthroughs while R5 is pure loss.

---

## Testing Strategy

### Unit Tests for Each Component

```python
# test_inheritance.py
def test_worst_inheritance_positive_trait():
    """First-born gets min of positive trait."""
    config = ExperimentConfig()
    engine = InheritanceEngine(config)
    
    p1_traits = np.array([0.8] * 15)  # High everything
    p2_traits = np.array([0.3] * 15)  # Low everything
    
    # Conscientiousness (index 1) has desirability +1
    child = engine.inherit(p1, p2, birth_order=1, population=[])
    
    # First-born should get WORST = min for positive trait
    assert child[TraitSystem.CONSCIENTIOUSNESS] == pytest.approx(0.3, abs=0.1)

def test_weirdest_inheritance():
    """Second-born gets trait farther from population mean."""
    # ... test weirdest logic
```

### Integration Tests

```python
def test_full_generation_runs():
    """Simulation can run a full generation without errors."""
    config = ExperimentConfig(initial_population=20)
    engine = SimulationEngine(config, ExtensionRegistry())
    metrics = engine.run(generations=1)
    
    assert metrics.history[0].population_size > 0

def test_trait_drift_over_generations():
    """Traits change over time when drift is enabled."""
    config = ExperimentConfig(trait_drift_rate=0.1)
    # ... run simulation, check that traits changed
```

### Property-Based Tests

```python
from hypothesis import given, strategies as st

@given(st.floats(0, 1), st.floats(0, 1))
def test_attraction_is_bounded(trait1, trait2):
    """Attraction scores should always be in [0, 1]."""
    # ...
```

---

## Example Usage

### Basic Simulation

```python
from seldon.core.config import ExperimentConfig
from seldon.core.engine import SimulationEngine
from seldon.extensions.registry import ExtensionRegistry

# Create config with custom parameters
config = ExperimentConfig(
    experiment_name="my_first_run",
    initial_population=100,
    generations_to_run=50,
    birth_order_rules={1: "worst", 2: "weirdest", 3: "best"},
    trait_drift_rate=0.02,
)

# Run simulation
engine = SimulationEngine(config, ExtensionRegistry())
metrics = engine.run(config.generations_to_run)

# Analyze results
print(f"Final population: {metrics.history[-1].population_size}")
print(f"Total breakthroughs: {sum(m.breakthrough_count for m in metrics.history)}")
```

### A/B Testing

```python
from seldon.experiment.runner import ExperimentRunner
from seldon.experiment.presets import config_baseline, config_no_birth_order

runner = ExperimentRunner()

# Run both configurations
runner.run_experiment(config_baseline())
runner.run_experiment(config_no_birth_order())

# Compare breakthrough rates
comparison = runner.compare_experiments(
    ['baseline', 'no_birth_order'],
    'breakthrough_count'
)
```

### Parameter Sweep

```python
# Test different sacrificial thresholds
for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
    config = ExperimentConfig(
        experiment_name=f"sacrificial_{threshold}",
        region_thresholds={
            'under_to_optimal': 0.3,
            'optimal_to_deep': 0.5,
            'deep_to_extreme': threshold,
            'productive_potential_threshold': 0.5,
        }
    )
    runner.run_experiment(config)

# Compare all runs
results = runner.compare_experiments(
    [f"sacrificial_{t}" for t in [0.5, 0.6, 0.7, 0.8, 0.9]],
    'region_sacrificial'
)
```

---

## Dependencies

```
# requirements.txt
numpy>=1.24.0
dataclasses-json>=0.6.0  # For config serialization
pytest>=7.0.0
pytest-cov>=4.0.0
hypothesis>=6.0.0  # Property-based testing

# Optional for visualization (Phase 2)
matplotlib>=3.7.0
pandas>=2.0.0
plotly>=5.0.0

# Optional for notebooks
jupyter>=1.0.0
```

---

## Questions to Resolve During Implementation

1. **Pairing duration** — Do partners stay together forever, or can relationships end?
   - Suggestion: Start with permanent pairing, add dissolution as config option later

2. **Multiple children per generation** — Can a pair have multiple children in one generation?
   - Architecture suggests `max_children_per_year: int = 1` in config

3. **Death of partner** — What happens when a partner dies? Can agent re-pair?
   - Suggestion: Yes, allow re-pairing after partner death

4. **Contribution calculation** — How exactly is contribution calculated from traits + region?
   - Need to implement: base contribution from region_effects, modified by creativity/resilience

5. **Breakthrough mechanics** — What happens when a breakthrough occurs?
   - Currently: just increments counter and tech_level
   - Could add: breakthrough benefits whole population, named discoveries, etc.

---

## Context from Project Creator

This project was designed by Kevin Keller of Tucuxi Inc. Key context:

1. **RSH Background** — The Reasoning Saturation Hypothesis is Kevin's research on cognitive processing patterns. The Five Regions model (Under-Processing → Optimal → Deep → Sacrificial → Pathological) comes from this work.

2. **Experimentation Focus** — This is NOT meant to simulate "reality" — it's meant to explore "what if?" questions. The birth order rules (1st=worst, 2nd=weirdest, 3rd=best) are hypotheses to test, not facts.

3. **No Equilibrium Targets** — We're not trying to maintain any particular distribution. We want to see what emerges from different parameter settings.

4. **LLM for Interviews Only** — The simulation runs on pure math. LLMs are only used for agent interviews and high-stakes decision reasoning (build this last).

5. **Visualization Matters** — Being able to see time series of metrics, trait distributions, and region populations is essential for understanding what's happening.

---

## Getting Started Checklist

- [ ] Create project structure
- [ ] Set up pyproject.toml / requirements.txt
- [ ] Implement TraitSystem (15 traits)
- [ ] Implement ExperimentConfig with all parameters
- [ ] Implement Agent dataclass
- [ ] Implement ProcessingRegion enum and ProcessingClassifier
- [ ] Implement InheritanceEngine with configurable birth order
- [ ] Implement TraitDriftEngine
- [ ] Implement AttractionModel
- [ ] Implement basic SimulationEngine loop
- [ ] Implement MetricsCollector
- [ ] Write tests for each component
- [ ] Create basic example script
- [ ] Run first simulation!

---

*Handoff Document Version: 1.0*
*Architecture Version: 2.3*
*Last Updated: January 2025*

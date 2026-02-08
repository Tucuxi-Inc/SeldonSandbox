# The Seldon Sandbox: A Psychohistory Engine
## Master Architecture Document v3.0

---

## Executive Summary

The Seldon Sandbox is an **experimentation platform** for multi-generational societal simulation. Every aspect of the system is exposed as a tunable parameter to explore "what if?" questions about community composition, agent orchestration, and emergent group dynamics.

The deeper purpose: understanding how to compose teams and communities of agents with complementary traits — what mix of cognitive processing styles optimizes group output, how birth-order-driven inheritance affects community resilience, what personality compositions make settlements succeed or fail, and how cultural memes sustain or destabilize populations.

### Version 3.0 Changes (from v2.3)

- **Configurable Trait System** — Trait count is no longer hardcoded at 15. Ships with two presets: Compact (15 traits) and Full (50 traits). Custom trait sets supported.
- **Utility-Based Decision Model** — All agent decisions use a unified mathematical framework: `U(a|P,x) = P^T · W_a · x + b_a` with softmax action selection. No more scattered threshold checks.
- **Cognitive Council** — Lightweight internal sub-agent model where 8 cognitive voices vote on decisions, weighted by personality traits.
- **Lore & Memory System** — Elevated from "future extension" to Phase 2 core. Generational memory with fidelity decay (the "Telephone Game" mechanic).
- **Outsider Interface** — Mid-simulation agent injection with ripple effect tracking.
- **Archetype Seed Vectors** — 11 pre-built personality templates (Da Vinci, Einstein, Curie, etc.) as founding population presets.
- **Narrative Visualization Architecture** — Story-driven visualization system beyond raw data charts.
- **Revised Extension Priority** — Migration/composition analysis elevated; it's the most relevant feature for agent orchestration research.
- **Social Dynamics** — Infidelity, non-traditional families, relationship dissolution, and societal pressure as configurable parameters.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           EXPERIMENT RUNNER                                  │
│              (A/B testing, parameter sweeps, archetype experiments)           │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                          EXPERIMENT CONFIG                                    │
│          (All parameters as tunable sliders — nothing hardcoded)              │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
         ┌──────────────────────────┼──────────────────────────┐
         ▼                          ▼                          ▼
┌────────────────────┐   ┌────────────────────┐   ┌────────────────────┐
│    CORE ENGINE     │   │     EXTENSIONS     │   │  METRICS & VIZ     │
│                    │   │    (Optional)      │   │                    │
│ • TraitSystem (N)  │   │ • Geography        │   │ • Time series      │
│ • Decision Model   │   │ • Resources        │   │ • Narrative views   │
│ • Cognitive Council│   │ • Migration        │   │ • Run comparison   │
│ • Inheritance      │   │ • Conflict         │   │ • Agent explorer   │
│ • Processing (RSH) │   │ • Culture/Memes    │   │ • Settlement diag. │
│ • Attraction       │   │ • Technology       │   │ • Family trees     │
│ • Reproduction     │   │ • Lore/Memory      │   │                    │
│ • Mortality        │   │                    │   │                    │
│ • Outsider Inject  │   │                    │   │                    │
└────────────────────┘   └────────────────────┘   └────────────────────┘
```

---

## Part 1: Configurable Trait System

### 1.1 Design Principle

The trait system is **not hardcoded to a specific dimensionality**. The `TraitSystem` class reads trait definitions from configuration, enabling different levels of personality modeling:

- **Compact (15 traits)** — Fast simulation, covers the essentials
- **Full (50 traits)** — Rich personality modeling from the ChatGPT-developed taxonomy
- **Custom** — User-defined trait sets for specialized experiments

### 1.2 Compact Trait Set (15 traits)

Default for quick experiments and core development:

| Index | Trait | Desirability | Description |
|-------|-------|-------------|-------------|
| 0 | Openness | +1 | Curiosity, creativity, willingness to explore |
| 1 | Conscientiousness | +1 | Organization, discipline, reliability |
| 2 | Extraversion | 0 | Social energy (neutral — neither good nor bad) |
| 3 | Agreeableness | +1 | Cooperation, trust, empathy |
| 4 | Neuroticism | -1 | Anxiety, emotional instability |
| 5 | Creativity | +1 | Novel idea generation |
| 6 | Resilience | +1 | Recovery from adversity |
| 7 | Ambition | 0 | Drive to achieve (neutral) |
| 8 | Empathy | +1 | Understanding others' emotions |
| 9 | Dominance | 0 | Leadership assertion (neutral) |
| 10 | Trust | +1 | Willingness to rely on others |
| 11 | Risk-Taking | 0 | Tolerance for uncertainty (neutral) |
| 12 | Adaptability | +1 | Flexibility in changing circumstances |
| 13 | Self-Control | +1 | Impulse regulation |
| 14 | Depth Drive | 0 | Tendency toward deep processing (key for RSH) |

### 1.3 Full Trait Set (50 traits)

Derived from the ChatGPT conversation's rigorous taxonomy, organized into functional groups:

**Core Domains (0-4):** Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism

**Cognitive & Innovation (5-11):** Curiosity, Creativity, Innovativeness, Intellectual Curiosity, Systematic Thinking, Open-Mindedness, Resourcefulness

**Adaptation & Resilience (12-15):** Adaptability, Resilience, Perseverance, Focus

**Social & Interpersonal (16-22):** Empathy, Assertiveness, Altruism, Tolerance, Trust, Sociability, Collaboration

**Self-Regulation & Drive (23-31):** Patience, Self-Efficacy, Optimism, Ambition, Confidence, Self-Control, Decisiveness, Integrity, Humility

**Emotional Processing (32-37):** Emotional Stability, Emotional Expressiveness, Reflectiveness, Self-Awareness, Empathic Accuracy, Mindfulness

**Pragmatic & Style (38-49):** Risk-Taking, Pragmatism, Independence, Competitiveness, Detail Orientation, Big-Picture Thinking, Enthusiasm, Humor, Caution, Boldness, Altruistic Leadership, Ethical Reasoning

Each trait has a configurable **desirability value** (-1, 0, or +1) that the inheritance engine uses for "worst"/"best" calculations, and a **stability value** (0-1) controlling how resistant it is to drift.

### 1.4 TraitSystem Implementation

```python
class TraitSystem:
    """
    Configurable trait definitions.
    Trait count is determined by the active trait preset, not hardcoded.
    """

    def __init__(self, config: 'ExperimentConfig'):
        self.preset = config.trait_preset  # 'compact', 'full', or 'custom'
        self.traits = self._load_trait_definitions(config)
        self.count = len(self.traits)

        # Build index constants dynamically
        for i, trait_def in enumerate(self.traits):
            setattr(self, trait_def['name'].upper().replace('-', '_').replace(' ', '_'), i)

    @property
    def desirability(self) -> np.ndarray:
        """Desirability vector for inheritance calculations."""
        return np.array([t['desirability'] for t in self.traits])

    @property
    def stability(self) -> np.ndarray:
        """Stability vector for drift calculations."""
        return np.array([t['stability'] for t in self.traits])

    def trait_name(self, index: int) -> str:
        return self.traits[index]['name']

    def trait_index(self, name: str) -> int:
        for i, t in enumerate(self.traits):
            if t['name'].lower() == name.lower():
                return i
        raise KeyError(f"Unknown trait: {name}")

    def random_traits(self, rng: np.random.Generator = None) -> np.ndarray:
        """Generate a random trait vector respecting configured distributions."""
        rng = rng or np.random.default_rng()
        return rng.uniform(
            low=[t.get('min', 0.0) for t in self.traits],
            high=[t.get('max', 1.0) for t in self.traits],
        )
```

### 1.5 Archetype Seed Vectors

11 pre-built personality templates for founding population experiments. Each is a complete trait vector derived from the documented behaviors and narratives of historical/fictional figures:

| # | Archetype | Key Traits | Use Case |
|---|-----------|-----------|----------|
| 1 | Leonardo da Vinci | Very high openness, creativity, curiosity; moderate conscientiousness | Creative explorer societies |
| 2 | Albert Einstein | Very high intellectual curiosity, openness; moderate social traits | Innovation-focused populations |
| 3 | Maria Montessori | High empathy, patience, conscientiousness; high openness | Education-oriented communities |
| 4 | Socrates | Very high reflectiveness, openness; moderate agreeableness | Philosophical societies |
| 5 | Marie Curie | Very high perseverance, focus, resilience; high depth drive | Research-intensive populations |
| 6 | Fred Rogers | Very high empathy, agreeableness, patience; moderate ambition | Harmony-focused communities |
| 7 | John Dewey | High systematic thinking, openness; balanced social traits | Democratic societies |
| 8 | Albus Dumbledore | High wisdom traits, leadership; moderate risk-taking | Guided leadership models |
| 9 | Yoda | Very high mindfulness, patience, self-control; low impulsivity | Wisdom-oriented populations |
| 10 | Ada Lovelace | High systematic thinking, creativity, vision; moderate extraversion | Technical innovation societies |
| 11 | Carl Sagan | High curiosity, communication, openness; moderate ambition | Exploration-driven communities |

Archetypes include full 50-dimensional vectors (from the ChatGPT document) and auto-projected 15-dimensional versions for compact mode.

```python
# Example: Start a society from 6 Einsteins and 5 Curies
config = ExperimentConfig(
    founding_population='archetypes',
    founding_archetypes=[
        ('einstein', 6),
        ('curie', 5),
    ],
    founding_trait_noise=0.1,  # Add variation so they're not identical
)
```

---

## Part 2: Utility-Based Decision Model

### 2.1 Core Principle

Every agent decision — pairing, reproduction, migration, conflict response — uses a **unified mathematical framework** rather than scattered threshold checks:

```
U(action | personality, context) = personality^T · W_action · context + b_action
```

Where:
- `personality` (P) = agent's trait vector, shape (N,)
- `context` (x) = situation vector (available partners, resource levels, threat level, etc.)
- `W_action` = learned/configured weight matrix per action type, shape (N, context_dim)
- `b_action` = bias term per action
- Action selection via softmax: `Pr(action) = exp(U) / sum(exp(U'))`

### 2.2 Why This Matters

This separates **"who someone is"** (their personality) from **"what they do"** in a given context. The same personality can produce different behaviors in different situations, and the decision is always explainable — you can inspect per-trait contributions to understand *why* an agent chose an action.

### 2.3 Decision Contexts

```python
class DecisionContext(Enum):
    PAIRING = "pairing"           # Choose a partner
    REPRODUCTION = "reproduction" # Whether to have children
    MIGRATION = "migration"       # Whether to leave, where to go
    CONFLICT = "conflict"         # How to respond to conflict
    CONTRIBUTION = "contribution" # How much effort to invest
    SOCIAL = "social"             # Social interactions and alliances
```

### 2.4 Implementation

```python
class DecisionModel:
    """
    Unified utility-based decision engine.

    All agent choices flow through this model:
    1. Build context vector from situation
    2. Compute utility for each available action
    3. Select action via softmax (with temperature from config)
    4. Return action + explainability data (per-trait contributions)
    """

    def __init__(self, config: 'ExperimentConfig'):
        self.config = config
        self.temperature = config.decision_temperature  # Controls randomness
        # Weight matrices loaded from config, one per decision context
        self.weights: Dict[DecisionContext, np.ndarray] = config.decision_weights
        self.biases: Dict[DecisionContext, np.ndarray] = config.decision_biases

    def decide(self, agent: 'Agent', context: DecisionContext,
               situation: np.ndarray, actions: List[str]) -> DecisionResult:
        """
        Compute utility for each action and select one.

        Returns DecisionResult with:
        - chosen_action: str
        - probabilities: dict of action -> probability
        - contributions: dict of trait_name -> influence on decision (explainability)
        """
        W = self.weights[context]
        b = self.biases[context]

        # Compute utilities for each action
        utilities = {}
        for i, action in enumerate(actions):
            # U = P^T · W_a · x + b_a
            utilities[action] = agent.traits @ W[i] @ situation + b[i]

        # Softmax selection
        values = np.array(list(utilities.values()))
        exp_values = np.exp((values - values.max()) / self.temperature)
        probabilities = exp_values / exp_values.sum()

        # Weighted random selection
        chosen_idx = np.random.choice(len(actions), p=probabilities)

        # Explainability: per-trait contribution to chosen action
        contributions = agent.traits * (W[chosen_idx] @ situation)

        return DecisionResult(
            chosen_action=actions[chosen_idx],
            probabilities=dict(zip(actions, probabilities)),
            trait_contributions=contributions,
        )
```

---

## Part 3: Cognitive Council

### 3.1 Concept

From the Gemini conversation: each agent has 8 internal cognitive "voices" that vote on decisions. Their influence is weighted by the agent's personality traits. This adds a layer of psychological realism — the same decision can be contested internally, and personality determines which voice wins.

### 3.2 The Eight Sub-Agents

| Sub-Agent | Role | Dominant Trait Weights |
|-----------|------|----------------------|
| **Cortex** | Rational analysis, central coordinator | Conscientiousness, Systematic Thinking |
| **Seer** | Visionary, anticipates future | Openness, Creativity, Big-Picture Thinking |
| **Oracle** | Deep analysis, counterpoint to Seer | Reflectiveness, Self-Awareness, Depth Drive |
| **House** | Pragmatic, day-to-day logistics | Pragmatism, Self-Control, Conscientiousness |
| **Prudence** | Cautious planning, risk mitigation | Caution, Conscientiousness, Self-Control |
| **Hypothalamus** | Primal emotional reactions | Neuroticism, Emotional Expressiveness, Risk-Taking |
| **Amygdala** | Emotional memory, social dynamics | Empathy, Trust, Agreeableness |
| **Conscience** | Moral judgment, ethical reasoning | Integrity, Altruism, Ethical Reasoning |

### 3.3 Voting Mechanism

```python
class CognitiveCouncil:
    """
    Lightweight internal decision model.

    Each sub-agent evaluates a decision and votes.
    Votes are weighted by personality traits.
    The council output modulates the utility-based decision model.

    Formula: final_action = sum(sub_agent_vote_i × trait_weight_i) / sum(trait_weight_i)
    """

    SUB_AGENTS = ['cortex', 'seer', 'oracle', 'house',
                  'prudence', 'hypothalamus', 'amygdala', 'conscience']

    def __init__(self, config: 'ExperimentConfig'):
        self.config = config
        self.enabled = config.cognitive_council_enabled
        # Maps sub-agent name -> list of (trait_index, weight) pairs
        self.trait_weights = config.cognitive_council_weights

    def compute_council_modulation(self, agent: 'Agent',
                                    context: DecisionContext) -> np.ndarray:
        """
        Compute how the cognitive council modulates decision utilities.

        Returns a modulation vector that adjusts utility scores.
        When council is disabled, returns ones (no modulation).
        """
        if not self.enabled:
            return np.ones(1)

        votes = {}
        for sub_agent in self.SUB_AGENTS:
            # Each sub-agent's influence = sum of its trait weights × agent's trait values
            influence = 0.0
            for trait_idx, weight in self.trait_weights[sub_agent]:
                influence += agent.traits[trait_idx] * weight

            # Sub-agent evaluates the context through its lens
            votes[sub_agent] = self._evaluate(sub_agent, context, influence)

        return self._aggregate_votes(votes)

    def get_dominant_voice(self, agent: 'Agent') -> str:
        """Which sub-agent has the strongest influence for this agent?"""
        strengths = {}
        for sub_agent in self.SUB_AGENTS:
            strengths[sub_agent] = sum(
                agent.traits[idx] * w
                for idx, w in self.trait_weights[sub_agent]
            )
        return max(strengths, key=strengths.get)
```

### 3.4 Integration with Decision Model

The cognitive council is an **optional modulation layer** that sits between raw utility computation and action selection:

```
Raw Utility → Council Modulation → Temperature Scaling → Softmax → Action
```

When `config.cognitive_council_enabled = False`, the council step is identity (no effect). This keeps the core decision model clean while allowing richer behavior when desired.

---

## Part 4: Core Agent Model

```python
@dataclass
class Agent:
    """Core agent — extensible, history-tracking, decision-capable."""
    id: str
    name: str
    age: int
    generation: int
    birth_order: int

    # Core traits (N-dimensional, where N = TraitSystem.count)
    traits: np.ndarray
    traits_at_birth: np.ndarray

    # Processing region (RSH Five Regions)
    processing_region: ProcessingRegion

    # State
    suffering: float = 0.0
    burnout_level: float = 0.0
    is_alive: bool = True

    # Relationships
    partner_id: Optional[str] = None
    parent1_id: Optional[str] = None
    parent2_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    relationship_status: str = "single"  # single, paired, widowed, dissolved

    # Social dynamics
    infidelity_history: List[dict] = field(default_factory=list)
    social_bonds: Dict[str, float] = field(default_factory=dict)  # agent_id -> bond_strength

    # Cognitive council state (which voice is dominant right now)
    dominant_voice: Optional[str] = None

    # Decision history (for explainability)
    decision_history: List['DecisionResult'] = field(default_factory=list)

    # History (for visualization — append each generation)
    trait_history: List[np.ndarray] = field(default_factory=list)
    region_history: List[ProcessingRegion] = field(default_factory=list)
    contribution_history: List[float] = field(default_factory=list)
    suffering_history: List[float] = field(default_factory=list)

    # Lore/memory (for generational memory system)
    personal_memories: List['Memory'] = field(default_factory=list)
    inherited_lore: List['Memory'] = field(default_factory=list)

    # === EXTENSION HOOKS ===
    location_id: Optional[str] = None
    resource_holdings: Dict[str, float] = field(default_factory=dict)
    cultural_memes: List[str] = field(default_factory=list)
    skills: Dict[str, float] = field(default_factory=dict)
    extension_data: Dict[str, Any] = field(default_factory=dict)

    # Outsider tracking
    is_outsider: bool = False
    outsider_origin: Optional[str] = None
    injection_generation: Optional[int] = None
```

---

## Part 5: RSH Five Regions Processing Model

### 5.1 The Five Regions

From Kevin's Reasoning Saturation Hypothesis paper:

| Region | Code | Key Behavior | Output | Suffering |
|--------|------|-------------|--------|-----------|
| R1: Under-Processing | `UNDER_PROCESSING` | Quick, shallow decisions | Low | Low |
| R2: Optimal | `OPTIMAL` | Balanced, efficient processing | Peak sustainable | Low |
| R3: Deep | `DEEP` | Thorough but costly processing | High, recoverable | Moderate |
| R4: Sacrificial | `SACRIFICIAL` | Obsessive, productive suffering | Breakthroughs | High |
| R5: Pathological | `PATHOLOGICAL` | Obsessive, unproductive suffering | None (pure loss) | Very High |

**The R4 vs R5 distinction is critical**: both involve suffering, but R4 produces breakthroughs while R5 is pure loss. The `productive_potential_threshold` config parameter determines where the line falls.

### 5.2 Classification

```python
class ProcessingClassifier:
    """
    Classifies agents into RSH processing regions.
    All thresholds from config — never hardcoded.

    Uses depth_drive trait + burnout + suffering to determine region.
    Transitions are possible each generation based on trait drift and state changes.
    """

    def classify(self, agent: Agent, config: ExperimentConfig) -> ProcessingRegion:
        trait_system = config.trait_system
        depth = agent.traits[trait_system.trait_index('depth_drive')]
        thresholds = config.region_thresholds

        if depth < thresholds['under_to_optimal']:
            return ProcessingRegion.UNDER_PROCESSING
        elif depth < thresholds['optimal_to_deep']:
            return ProcessingRegion.OPTIMAL
        elif depth < thresholds['deep_to_extreme']:
            return ProcessingRegion.DEEP
        else:
            # R4 vs R5: productive potential determines which
            potential = self._calculate_productive_potential(agent, config)
            if potential >= thresholds['productive_potential_threshold']:
                return ProcessingRegion.SACRIFICIAL
            else:
                return ProcessingRegion.PATHOLOGICAL

    def _calculate_productive_potential(self, agent: Agent,
                                         config: ExperimentConfig) -> float:
        """
        Determine if deep processing is productive (R4) or destructive (R5).
        Based on creativity, resilience, and burnout level.
        """
        ts = config.trait_system
        creativity = agent.traits[ts.trait_index('creativity')]
        resilience = agent.traits[ts.trait_index('resilience')]

        potential = (creativity * config.productive_weights['creativity'] +
                     resilience * config.productive_weights['resilience'] -
                     agent.burnout_level * config.productive_weights['burnout_penalty'])

        return np.clip(potential, 0.0, 1.0)
```

---

## Part 6: Inheritance Engine

### 6.1 Birth Order Rules

The foundational hypothesis driving the simulation:

| Birth Order | Rule | Mechanism |
|-------------|------|-----------|
| 1st | `"worst"` | Gets the less desirable value per trait (using desirability map) |
| 2nd | `"weirdest"` | Gets whichever parent's value is farther from population mean |
| 3rd | `"best"` | Gets the more desirable value per trait |
| 4th+ | `"random_weighted"` | Weighted random mix of both parents |

**Important edge cases** (from the Gemini conversation):
- Dead children still count for birth-order assignment
- Identical twins share birth-order traits; fraternal twins get distinct positions
- All rules are configurable per position — testing inverted or custom rules is the whole point

### 6.2 Implementation

```python
class InheritanceEngine:
    """
    Birth-order-based trait inheritance.
    Rules are configurable per birth order position via config.
    """

    RULE_TYPES = {
        'worst': '_inherit_worst',
        'best': '_inherit_best',
        'weirdest': '_inherit_weirdest',
        'random_weighted': '_inherit_random_weighted',
        'average': '_inherit_average',
    }

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.trait_system = config.trait_system

    def inherit(self, parent1: Agent, parent2: Agent,
                birth_order: int, population: List[Agent],
                rng: np.random.Generator = None) -> np.ndarray:
        """
        Generate child traits based on parents and birth order.

        1. Look up rule for this birth order position
        2. Apply rule to get base traits
        3. Add environmental noise (Gaussian)
        4. Clamp to [0, 1]
        """
        rng = rng or np.random.default_rng()
        rule = self.config.birth_order_rules.get(birth_order, 'random_weighted')

        method = getattr(self, self.RULE_TYPES[rule])
        base_traits = method(parent1.traits, parent2.traits, population)

        # Environmental noise: epsilon ~ N(0, sigma^2)
        noise = rng.normal(0, self.config.inheritance_noise_sigma,
                           size=self.trait_system.count)

        return np.clip(base_traits + noise, 0.0, 1.0)

    def _inherit_worst(self, p1_traits, p2_traits, population):
        """First-born: gets less desirable value per trait."""
        desirability = self.trait_system.desirability
        result = np.empty(self.trait_system.count)
        for i in range(self.trait_system.count):
            if desirability[i] > 0:      # Positive trait → take minimum
                result[i] = min(p1_traits[i], p2_traits[i])
            elif desirability[i] < 0:    # Negative trait → take maximum
                result[i] = max(p1_traits[i], p2_traits[i])
            else:                         # Neutral → average
                result[i] = (p1_traits[i] + p2_traits[i]) / 2
        return result

    def _inherit_weirdest(self, p1_traits, p2_traits, population):
        """Second-born: gets trait farther from population mean."""
        if not population:
            return (p1_traits + p2_traits) / 2

        pop_mean = np.mean([a.traits for a in population], axis=0)
        result = np.empty(self.trait_system.count)
        for i in range(self.trait_system.count):
            dist_p1 = abs(p1_traits[i] - pop_mean[i])
            dist_p2 = abs(p2_traits[i] - pop_mean[i])
            result[i] = p1_traits[i] if dist_p1 >= dist_p2 else p2_traits[i]
        return result

    def _inherit_best(self, p1_traits, p2_traits, population):
        """Third-born: gets more desirable value per trait."""
        desirability = self.trait_system.desirability
        result = np.empty(self.trait_system.count)
        for i in range(self.trait_system.count):
            if desirability[i] > 0:
                result[i] = max(p1_traits[i], p2_traits[i])
            elif desirability[i] < 0:
                result[i] = min(p1_traits[i], p2_traits[i])
            else:
                result[i] = (p1_traits[i] + p2_traits[i]) / 2
        return result

    def _inherit_random_weighted(self, p1_traits, p2_traits, population,
                                  rng=None):
        """Fourth+: weighted random mix of both parents."""
        rng = rng or np.random.default_rng()
        weights = rng.uniform(0, 1, size=self.trait_system.count)
        return p1_traits * weights + p2_traits * (1 - weights)

    def _inherit_average(self, p1_traits, p2_traits, population):
        """Simple average of both parents."""
        return (p1_traits + p2_traits) / 2
```

---

## Part 7: Social Dynamics

### 7.1 Relationship Model

Relationships are richer than simple permanent pairing:

```python
# All parameters from config
relationship_config = {
    # Pairing
    'pairing_min_age': 16,
    'pairing_permanent': False,             # If True, no dissolution
    'dissolution_enabled': True,
    'dissolution_compatibility_threshold': 0.3,  # Below this, risk of split
    'dissolution_base_rate': 0.05,          # Annual base rate

    # Re-pairing
    'reparing_after_death': True,
    'reparing_after_dissolution': True,
    'reparing_cooldown_generations': 1,

    # Infidelity
    'infidelity_enabled': True,
    'infidelity_base_rate': 0.15,           # 15-30% from research
    'infidelity_trait_modifiers': {
        'agreeableness': -0.3,              # High agreeableness reduces
        'conscientiousness': -0.2,          # High conscientiousness reduces
        'neuroticism': 0.2,                 # High neuroticism increases
        'risk_taking': 0.15,               # High risk-taking increases
    },
    'infidelity_compatibility_modifier': -0.5,  # Low compatibility increases

    # Non-traditional families
    'single_by_choice_rate': 0.10,          # 5-15% remain single for life
    'lgbtq_rate': 0.035,                    # 2-5% population
    'assisted_reproduction_rate': 0.075,    # 5-10% use assisted reproduction
}
```

### 7.2 Fertility Constraints

From the Claude simulation results:

```python
fertility_config = {
    # Biological constraints
    'female_fertility_start': 16,
    'female_fertility_end': 40,
    'male_fertility_start': 16,
    'male_fertility_end': None,             # No hard limit

    # Birth spacing
    'min_birth_spacing_generations': 1,
    'max_children_per_generation': 1,

    # Mortality
    'maternal_mortality_rate': 0.015,       # 1-2% per birth
    'child_mortality_rate': 0.30,           # 25-35%

    # Societal pressure
    'societal_fertility_pressure': 0.5,     # Pressure to reproduce (0=none, 1=strong)
    'target_children_mean': 3.0,            # Societal norm for family size
}
```

---

## Part 8: Lore & Memory System

### 8.1 Concept

From the Gemini conversation's "Telephone Game" mechanic: factual events degrade into myths over ~20 generations. Personal memories are accurate; inherited lore has fidelity decay. This directly affects agent decision-making — agents act on what they *believe* happened, not what actually happened.

### 8.2 Memory Types

```python
@dataclass
class Memory:
    """A piece of information an agent holds."""
    id: str
    content: str                          # What the memory is about
    memory_type: MemoryType               # PERSONAL, FAMILY, SOCIETAL, MYTH
    original_event_id: Optional[str]      # Links back to actual simulation event
    generation_created: int               # When this memory was first formed
    generation_inherited: Optional[int]   # When this agent received it

    # Fidelity tracking
    fidelity: float = 1.0                 # 1.0 = perfectly accurate, 0.0 = pure myth
    distortion_vector: np.ndarray = None  # How the memory has been altered

    # Behavioral effects
    trait_modifiers: Dict[str, float] = field(default_factory=dict)  # How this memory affects decisions
    emotional_valence: float = 0.0        # -1 (traumatic) to +1 (positive)


class MemoryType(Enum):
    PERSONAL = "personal"       # Direct experience — high fidelity
    FAMILY = "family"           # Passed from parent — moderate fidelity
    SOCIETAL = "societal"       # Community knowledge — lower fidelity
    MYTH = "myth"               # Ancient lore — heavily distorted
```

### 8.3 Fidelity Decay

```python
class LoreEngine:
    """
    Manages generational memory transmission and degradation.

    Key mechanics:
    - Personal memories: fidelity = 1.0, no decay
    - Each transmission: fidelity *= (1 - decay_rate)
    - Below myth_threshold: memory becomes MYTH type, content may mutate
    - Myths can merge, split, or invert over time
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.decay_rate = config.lore_decay_rate              # e.g., 0.05 per generation
        self.myth_threshold = config.lore_myth_threshold      # e.g., 0.3
        self.mutation_rate = config.lore_mutation_rate         # e.g., 0.02
        self.transmission_probability = config.lore_transmission_rate  # e.g., 0.7

    def transmit_to_child(self, parent: Agent, child: Agent,
                           rng: np.random.Generator = None):
        """
        Pass memories from parent to child with fidelity loss.

        Not all memories are transmitted — probability based on
        emotional salience and repetition.
        """
        rng = rng or np.random.default_rng()

        for memory in parent.personal_memories + parent.inherited_lore:
            # More emotionally salient memories more likely to be passed on
            transmission_prob = self.transmission_probability * (
                0.5 + 0.5 * abs(memory.emotional_valence)
            )

            if rng.random() < transmission_prob:
                inherited = self._copy_with_decay(memory, child.generation, rng)
                child.inherited_lore.append(inherited)

    def _copy_with_decay(self, memory: Memory, current_gen: int,
                          rng: np.random.Generator) -> Memory:
        """Copy memory with fidelity decay and possible mutation."""
        new_fidelity = memory.fidelity * (1 - self.decay_rate)

        # Determine new type based on fidelity
        if new_fidelity < self.myth_threshold:
            new_type = MemoryType.MYTH
        elif memory.memory_type == MemoryType.PERSONAL:
            new_type = MemoryType.FAMILY
        else:
            new_type = MemoryType.SOCIETAL

        # Possible mutation: emotional valence can drift, details can invert
        new_valence = memory.emotional_valence
        if rng.random() < self.mutation_rate:
            new_valence += rng.normal(0, 0.1)
            new_valence = np.clip(new_valence, -1.0, 1.0)

        return Memory(
            id=f"{memory.id}_gen{current_gen}",
            content=memory.content,
            memory_type=new_type,
            original_event_id=memory.original_event_id,
            generation_created=memory.generation_created,
            generation_inherited=current_gen,
            fidelity=new_fidelity,
            emotional_valence=new_valence,
            trait_modifiers=memory.trait_modifiers.copy(),
        )

    def evolve_societal_lore(self, population: List[Agent], generation: int):
        """
        Societal-level lore evolution:
        - Widely-held memories reinforce each other
        - Contradictory memories compete
        - Very old low-fidelity memories can merge into new myths
        """
        # Implementation: aggregate shared memories, resolve conflicts,
        # create consensus narratives
        pass
```

---

## Part 9: Outsider Interface

### 9.1 Concept

From the Gemini conversation: inject agents with custom traits into an existing population mid-simulation. Track the "ripple effect" — how the outsider's traits propagate through inheritance, affect attraction patterns, disrupt processing region distributions, and influence cultural dynamics.

### 9.2 Implementation

```python
class OutsiderInterface:
    """
    Mid-simulation agent injection with ripple effect tracking.

    Use cases:
    - "What happens when a disruptive personality enters an established community?"
    - "How quickly do outsider traits propagate through a population?"
    - "Do outsiders trigger migration or settlement founding?"
    """

    def inject_outsider(self, population: List[Agent],
                         traits: np.ndarray,
                         generation: int,
                         config: ExperimentConfig,
                         name: str = None,
                         origin: str = "external") -> Agent:
        """
        Inject a new agent into the population.

        The outsider has no parents in the simulation,
        custom traits (possibly outside normal population distribution),
        and is tracked separately for ripple effect analysis.
        """
        outsider = Agent(
            id=f"outsider_{generation}_{uuid4().hex[:8]}",
            name=name or f"Outsider-{generation}",
            age=config.outsider_injection_age,  # e.g., 20
            generation=generation,
            birth_order=0,  # N/A for outsiders
            traits=traits,
            traits_at_birth=traits.copy(),
            processing_region=ProcessingRegion.OPTIMAL,  # Will be reclassified
            is_outsider=True,
            outsider_origin=origin,
            injection_generation=generation,
        )

        population.append(outsider)
        return outsider

    def inject_archetype(self, population: List[Agent],
                          archetype_name: str,
                          generation: int,
                          config: ExperimentConfig,
                          noise: float = 0.05) -> Agent:
        """Inject an agent based on a named archetype."""
        base_traits = config.archetypes[archetype_name]
        rng = np.random.default_rng()
        traits = np.clip(base_traits + rng.normal(0, noise, size=len(base_traits)), 0, 1)
        return self.inject_outsider(population, traits, generation, config,
                                     name=f"{archetype_name.title()}-{generation}")


class RippleTracker:
    """
    Tracks how an outsider's traits propagate through the population.

    Metrics:
    - Trait diffusion: how outsider traits spread via inheritance
    - Pairing patterns: who pairs with outsiders or their descendants
    - Region shifts: processing region distribution changes after injection
    - Cultural impact: meme adoption changes
    - Lineage size: how many descendants after N generations
    """

    def __init__(self):
        self.injections: List[dict] = []
        self.snapshots: Dict[str, List[dict]] = {}  # outsider_id -> gen snapshots

    def track_injection(self, outsider: Agent, population: List[Agent]):
        """Record population state at injection time."""
        self.injections.append({
            'outsider_id': outsider.id,
            'injection_gen': outsider.injection_generation,
            'outsider_traits': outsider.traits.copy(),
            'pop_trait_mean': np.mean([a.traits for a in population], axis=0),
            'pop_trait_std': np.std([a.traits for a in population], axis=0),
            'pop_region_dist': self._region_distribution(population),
        })

    def track_generation(self, outsider_id: str, generation: int,
                          population: List[Agent]):
        """Track ripple metrics for one generation after injection."""
        descendants = self._find_descendants(outsider_id, population)

        snapshot = {
            'generation': generation,
            'descendant_count': len(descendants),
            'descendant_trait_mean': np.mean([d.traits for d in descendants], axis=0) if descendants else None,
            'pop_trait_mean': np.mean([a.traits for a in population], axis=0),
            'pop_region_dist': self._region_distribution(population),
        }

        self.snapshots.setdefault(outsider_id, []).append(snapshot)
```

---

## Part 10: Extension Architecture

### 10.1 Extension Interface

All extensions implement this interface (unchanged from v2.3):

```python
class SimulationExtension(ABC):
    """
    Base class for all simulation extensions.

    Extensions can:
    - Add parameters to ExperimentConfig
    - Hook into generation phases
    - Add metrics to track
    - Modify agent behavior via decision model adjustments
    """

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @abstractmethod
    def get_default_config(self) -> Dict[str, Any]:
        pass

    # Lifecycle hooks
    def on_simulation_start(self, population, config): pass
    def on_generation_start(self, generation, population, config): pass
    def on_agent_created(self, agent, parents, config): pass
    def on_agent_decision(self, agent, decision_context, config) -> dict:
        return decision_context
    def on_generation_end(self, generation, population, config): pass

    # Modifier hooks
    def modify_attraction(self, a1, a2, base, config) -> float:
        return base
    def modify_mortality(self, agent, base_rate, config) -> float:
        return base_rate
    def modify_decision(self, agent, context, utilities, config) -> dict:
        """NEW in v3.0: Modify utility scores before action selection."""
        return utilities

    # Metrics
    def get_metrics(self, population) -> Dict[str, Any]:
        return {}
```

### 10.2 Extension Registry

```python
class ExtensionRegistry:
    """Manages simulation extensions with dependency resolution."""

    def __init__(self):
        self._extensions: Dict[str, SimulationExtension] = {}
        self._enabled: Dict[str, bool] = {}

    def register(self, extension: SimulationExtension):
        self._extensions[extension.name] = extension
        self._enabled[extension.name] = False

    def enable(self, name: str):
        """Enable an extension, checking dependencies."""
        ext = self._extensions.get(name)
        if ext:
            deps = ext.get_default_config().get('requires', [])
            for dep in deps:
                if not self._enabled.get(dep, False):
                    raise ValueError(f"Extension '{name}' requires '{dep}' to be enabled first")
            self._enabled[name] = True

    def disable(self, name: str):
        self._enabled[name] = False

    def get_enabled(self) -> List[SimulationExtension]:
        return [ext for name, ext in self._extensions.items()
                if self._enabled.get(name, False)]

    def get_combined_config(self) -> Dict[str, Any]:
        return {ext.name: ext.get_default_config() for ext in self.get_enabled()}
```

### 10.3 Geography Extension

```python
class GeographyExtension(SimulationExtension):
    """
    Spatial dimension: hex grid or region-based map.
    Locations have carrying capacity; distance affects interaction.
    """

    name = "geography"
    description = "Spatial simulation with locations and movement"

    def get_default_config(self):
        return {
            'enabled': False,
            'map_type': 'hex',              # 'hex', 'regions', 'continuous'
            'map_size': (10, 10),
            'starting_locations': 3,
            'base_carrying_capacity': 50,
            'capacity_variation': 0.3,
            'max_interaction_distance': 2,
            'attraction_distance_decay': 0.5,
            'migration_enabled': True,
            'migration_threshold': 0.3,
        }

@dataclass
class Location:
    id: str
    name: str
    coordinates: Tuple[int, int]
    carrying_capacity: int = 50
    current_population: int = 0
    resource_richness: float = 1.0
    base_desirability: float = 0.5
    connections: List[str] = field(default_factory=list)
    connection_difficulties: Dict[str, float] = field(default_factory=dict)
```

### 10.4 Resources Extension

```python
class ResourcesExtension(SimulationExtension):
    """
    Resource scarcity dynamics: agents need resources to survive.
    Scarcity affects mortality, fertility, and conflict.
    """

    name = "resources"
    description = "Resource scarcity and economic dynamics"

    def get_default_config(self):
        return {
            'enabled': False,
            'resource_types': ['food', 'shelter', 'status'],
            'base_regeneration_rate': 0.1,
            'carrying_capacity_resource_multiplier': 1.0,
            'consumption_per_agent': {'food': 1.0, 'shelter': 0.5, 'status': 0.0},
            'scarcity_mortality_multiplier': 2.0,
            'scarcity_fertility_multiplier': 0.5,
            'scarcity_conflict_multiplier': 1.5,
            'resource_distribution': 'equal',
            'hoarding_enabled': True,
        }
```

### 10.5 Migration Extension

**Elevated priority in v3.0** — this is the most relevant extension for agent orchestration research, because the `evaluate_settlement_viability()` function directly models how group composition affects outcomes.

```python
class MigrationExtension(SimulationExtension):
    """
    Population movement and settlement founding.

    Key feature: settlement viability depends on group personality composition.
    This is the core agent orchestration experiment.
    """

    name = "migration"
    description = "Population movement and settlement founding"

    def get_default_config(self):
        return {
            'enabled': False,
            'requires': ['geography'],

            'push_factors': {
                'overcrowding_weight': 0.3,
                'scarcity_weight': 0.3,
                'conflict_weight': 0.2,
                'low_status_weight': 0.2,
            },
            'pull_factors': {
                'resources_weight': 0.3,
                'family_weight': 0.2,
                'opportunity_weight': 0.3,
                'safety_weight': 0.2,
            },

            'migration_decision_threshold': 0.6,
            'migration_cost': 0.2,

            'new_settlement_enabled': True,
            'min_founding_group_size': 5,
            'founding_difficulty': 0.7,

            # Composition effects — the core of agent orchestration research
            'settlement_composition_effects': {
                'min_conscientiousness_mean': 0.4,
                'max_neuroticism_mean': 0.7,
                'optimal_extraversion_variance': 0.2,
                'requires_leader': True,
                'min_region_diversity': 2,       # NEW: need multiple processing styles
                'optimal_r2_proportion': 0.4,    # NEW: need a core of optimal processors
                'min_r3r4_proportion': 0.1,      # NEW: need some deep/sacrificial for innovation
            },
        }

    def evaluate_settlement_viability(self, founding_group: List[Agent],
                                      config: ExperimentConfig) -> Tuple[float, List[str]]:
        """
        Evaluate whether a founding group can successfully start a settlement.
        Returns (success_probability, risk_factors).

        This is the core "agent orchestration" function — it answers:
        "Given this group of personalities, can they build something together?"
        """
        comp = config.extensions['migration']['settlement_composition_effects']
        success_prob = 1.0 - config.extensions['migration']['founding_difficulty']
        risks = []
        ts = config.trait_system

        traits = np.array([a.traits for a in founding_group])
        means = traits.mean(axis=0)
        stds = traits.std(axis=0)

        # Conscientiousness check (organization)
        if means[ts.trait_index('conscientiousness')] < comp['min_conscientiousness_mean']:
            success_prob *= 0.5
            risks.append("Low organization/conscientiousness")

        # Neuroticism check (collective anxiety)
        if means[ts.trait_index('neuroticism')] > comp['max_neuroticism_mean']:
            success_prob *= 0.6
            risks.append("High collective anxiety")

        # Leadership check
        if comp['requires_leader']:
            has_leader = any(
                a.traits[ts.trait_index('dominance')] > 0.7 for a in founding_group
            )
            if not has_leader:
                success_prob *= 0.4
                risks.append("No clear leader")

        # Extraversion mix
        extraversion_var = stds[ts.trait_index('extraversion')]
        if abs(extraversion_var - comp['optimal_extraversion_variance']) > 0.2:
            success_prob *= 0.8
            risks.append("Imbalanced social composition")

        # Processing region diversity (NEW in v3.0)
        regions = set(a.processing_region for a in founding_group)
        if len(regions) < comp.get('min_region_diversity', 2):
            success_prob *= 0.6
            risks.append("Insufficient cognitive diversity")

        # Need practical workers (R2) AND innovators (R3/R4)
        r2_count = sum(1 for a in founding_group if a.processing_region == ProcessingRegion.OPTIMAL)
        r3r4_count = sum(1 for a in founding_group if a.processing_region in
                         (ProcessingRegion.DEEP, ProcessingRegion.SACRIFICIAL))

        r2_prop = r2_count / len(founding_group)
        r3r4_prop = r3r4_count / len(founding_group)

        if r2_prop < comp.get('optimal_r2_proportion', 0.4):
            success_prob *= 0.7
            risks.append("Too few practical workers (R2)")

        if r3r4_prop < comp.get('min_r3r4_proportion', 0.1):
            success_prob *= 0.8
            risks.append("No innovators (R3/R4) in group")

        # All one type is bad
        if all(a.processing_region == ProcessingRegion.SACRIFICIAL for a in founding_group):
            success_prob *= 0.3
            risks.append("All deep processors, no practical workers")
        if all(a.processing_region == ProcessingRegion.UNDER_PROCESSING for a in founding_group):
            success_prob *= 0.7
            risks.append("All shallow processors, no depth")

        return (success_prob, risks)
```

### 10.6 Conflict Extension

```python
class ConflictExtension(SimulationExtension):
    """
    Personality-based conflict dynamics and resolution.
    Uses the decision model for conflict response selection.
    """

    name = "conflict"
    description = "Conflict dynamics and resolution"

    def get_default_config(self):
        return {
            'enabled': False,
            'triggers': {
                'resource_scarcity_threshold': 0.3,
                'dominance_clash_threshold': 0.8,
                'trust_betrayal_threshold': 0.2,
            },
            'trait_conflict_weights': {
                'dominance': 0.3,
                'agreeableness': -0.3,
                'neuroticism': 0.2,
            },
            'resolution_methods': ['submission', 'compromise', 'separation', 'escalation'],
            'resolution_trait_influences': {
                'submission': {'dominance': -0.3, 'agreeableness': 0.2},
                'compromise': {'agreeableness': 0.3, 'openness': 0.2},
                'separation': {'extraversion': -0.2},
                'escalation': {'dominance': 0.3, 'neuroticism': 0.2},
            },
            'conflict_suffering_cost': 0.2,
            'conflict_relationship_damage': 0.3,
        }
```

### 10.7 Culture/Memes Extension

```python
class CultureExtension(SimulationExtension):
    """
    Cultural meme propagation and evolution.

    From RSH: cultural memes like "tortured genius" maintain the ~5-10%
    sacrificial processing equilibrium. This extension tests that hypothesis.
    """

    name = "culture"
    description = "Cultural memes and their effects on behavior"

    def get_default_config(self):
        return {
            'enabled': False,
            'initial_memes': [
                {
                    'id': 'tortured_genius',
                    'description': 'Suffering is necessary for greatness',
                    'effects': {
                        'sacrificial_processing_bonus': 0.1,
                        'suffering_tolerance': 0.2,
                    },
                    'spread_rate': 0.1,
                    'initial_prevalence': 0.3,
                },
                {
                    'id': 'efficiency_worship',
                    'description': 'Optimal processing is the ideal',
                    'effects': {
                        'termination_ability_bonus': 0.1,
                        'depth_penalty': -0.1,
                    },
                    'spread_rate': 0.15,
                    'initial_prevalence': 0.5,
                },
                {
                    'id': 'family_first',
                    'description': 'Family and reproduction are primary',
                    'effects': {
                        'fertility_bonus': 0.2,
                        'ambition_penalty': -0.1,
                    },
                    'spread_rate': 0.12,
                    'initial_prevalence': 0.4,
                },
            ],
            'transmission_method': 'social_learning',
            'mutation_rate': 0.01,
            'extinction_threshold': 0.05,
        }
```

### 10.8 Technology Extension

```python
class TechnologyExtension(SimulationExtension):
    """
    Technological progress from R3/R4 breakthroughs.
    Tech level affects carrying capacity, mortality, and fertility.
    """

    name = "technology"
    description = "Technological advancement and its societal effects"

    def get_default_config(self):
        return {
            'enabled': False,
            'starting_tech_level': 1.0,
            'breakthrough_tech_increment': 0.1,
            'tech_capacity_multiplier': 1.5,
            'tech_mortality_reduction': 0.05,
            'tech_fertility_bonus': 0.02,
            'tech_decay_enabled': False,
            'tech_decay_rate': 0.01,
            'tech_maintenance_population': 50,
        }
```

---

## Part 11: Simulation Engine

### 11.1 Generation Loop with All Systems

```python
class SimulationEngine:
    """
    Main simulation loop integrating all v3.0 systems:
    - Configurable trait system
    - Utility-based decision model with cognitive council
    - Extension hooks at every phase
    - Outsider injection support
    - Lore transmission
    """

    def __init__(self, config: ExperimentConfig, extensions: ExtensionRegistry):
        self.config = config
        self.extensions = extensions

        # Core components
        self.trait_system = TraitSystem(config)
        self.inheritance = InheritanceEngine(config)
        self.classifier = ProcessingClassifier(config)
        self.drift_engine = TraitDriftEngine(config)
        self.attraction = AttractionModel(config)
        self.decision_model = DecisionModel(config)
        self.council = CognitiveCouncil(config)
        self.lore_engine = LoreEngine(config)
        self.outsider_interface = OutsiderInterface()
        self.ripple_tracker = RippleTracker()
        self.metrics = MetricsCollector(config)

    def run(self, generations: int) -> MetricsCollector:
        """Run the simulation."""
        population = self._create_initial_population()

        for ext in self.extensions.get_enabled():
            ext.on_simulation_start(population, self.config)

        for gen in range(generations):
            events = self._run_generation(gen, population)
            self.metrics.collect(gen, population, events)

            # Process any scheduled outsider injections
            self._process_scheduled_injections(gen, population)

            # Track ripple effects for any active outsiders
            for injection in self.ripple_tracker.injections:
                self.ripple_tracker.track_generation(
                    injection['outsider_id'], gen, population
                )

        return self.metrics

    def _run_generation(self, generation: int, population: List[Agent]) -> dict:
        events = {
            'births': 0, 'deaths': 0, 'breakthroughs': 0,
            'dissolutions': 0, 'migrations': 0, 'lore_events': 0,
        }

        # === Extension hook: generation start ===
        for ext in self.extensions.get_enabled():
            ext.on_generation_start(generation, population, self.config)

        # === Phase 1: Age and trait drift ===
        for agent in population:
            agent.age += 1
            experiences = self._get_agent_experiences(agent)
            agent.traits = self.drift_engine.drift_traits(agent, experiences)
            agent.traits = self.drift_engine.apply_region_effects(agent)
            agent.trait_history.append(agent.traits.copy())

        # === Phase 2: Processing region updates ===
        for agent in population:
            new_region = self.classifier.classify(agent, self.config)
            if new_region != agent.processing_region:
                events.setdefault('transitions', {})[
                    f"{agent.processing_region.value}_to_{new_region.value}"
                ] = events.get('transitions', {}).get(
                    f"{agent.processing_region.value}_to_{new_region.value}", 0
                ) + 1
            agent.processing_region = new_region
            agent.region_history.append(agent.processing_region)

            # Update dominant cognitive voice
            if self.config.cognitive_council_enabled:
                agent.dominant_voice = self.council.get_dominant_voice(agent)

        # === Phase 3: Contribution and breakthroughs ===
        for agent in population:
            contribution, breakthrough = self._calculate_contribution(agent)
            agent.contribution_history.append(contribution)
            if breakthrough:
                events['breakthroughs'] += 1
                # Create a memory of the breakthrough
                agent.personal_memories.append(Memory(
                    id=f"breakthrough_{generation}_{agent.id}",
                    content=f"Breakthrough in generation {generation}",
                    memory_type=MemoryType.PERSONAL,
                    generation_created=generation,
                    emotional_valence=0.9,
                ))

        # === Phase 4: Relationship dynamics ===
        # Check for dissolution of existing pairs
        if self.config.relationship_config.get('dissolution_enabled'):
            events['dissolutions'] = self._process_dissolutions(population)

        # Form new pairs via utility-based decision model
        eligible = [a for a in population
                    if a.age >= self.config.relationship_config['pairing_min_age']
                    and a.partner_id is None and a.is_alive]
        pairs = self._form_pairs(eligible)

        # === Phase 5: Reproduction ===
        new_children = []
        paired_agents = [(a, self._find_agent(a.partner_id, population))
                         for a in population if a.partner_id]

        for p1, p2 in paired_agents:
            if p2 and self._will_reproduce(p1, p2):
                child = self._create_child(p1, p2, generation, population)
                new_children.append(child)
                events['births'] += 1

                # Transmit lore to child
                self.lore_engine.transmit_to_child(p1, child)
                self.lore_engine.transmit_to_child(p2, child)

                for ext in self.extensions.get_enabled():
                    ext.on_agent_created(child, (p1, p2), self.config)

        population.extend(new_children)

        # === Phase 6: Lore evolution ===
        self.lore_engine.evolve_societal_lore(population, generation)

        # === Phase 7: Mortality ===
        for agent in population:
            base_rate = self._calculate_base_mortality(agent)
            for ext in self.extensions.get_enabled():
                base_rate = ext.modify_mortality(agent, base_rate, self.config)
            if np.random.random() < base_rate:
                agent.is_alive = False
                events['deaths'] += 1
                # Handle partner widowing
                if agent.partner_id:
                    partner = self._find_agent(agent.partner_id, population)
                    if partner:
                        partner.partner_id = None
                        partner.relationship_status = "widowed"

        population[:] = [a for a in population if a.is_alive]

        # === Extension hook: generation end ===
        for ext in self.extensions.get_enabled():
            ext.on_generation_end(generation, population, self.config)

        # Collect extension metrics
        for ext in self.extensions.get_enabled():
            events[f'ext_{ext.name}'] = ext.get_metrics(population)

        return events

    def _form_pairs(self, eligible: List[Agent]) -> List[Tuple[Agent, Agent]]:
        """Form pairs using utility-based decision model."""
        pairs = []
        unpaired = eligible.copy()

        while len(unpaired) >= 2:
            a1 = unpaired.pop(0)

            attractions = []
            for a2 in unpaired:
                base = self.attraction.calculate(a1, a2)

                # Cognitive council modulation
                if self.config.cognitive_council_enabled:
                    modulation = self.council.compute_council_modulation(
                        a1, DecisionContext.PAIRING
                    )
                    base *= modulation.mean()

                # Extension hooks
                for ext in self.extensions.get_enabled():
                    base = ext.modify_attraction(a1, a2, base, self.config)

                attractions.append((a2, base))

            if attractions:
                agents, scores = zip(*attractions)
                scores = np.array(scores)
                scores = np.maximum(scores, 0)
                if scores.sum() > 0:
                    probs = scores / scores.sum()
                    idx = np.random.choice(len(agents), p=probs)
                    partner = agents[idx]
                    unpaired.remove(partner)

                    a1.partner_id = partner.id
                    partner.partner_id = a1.id
                    a1.relationship_status = "paired"
                    partner.relationship_status = "paired"

                    pairs.append((a1, partner))

        return pairs
```

---

## Part 12: Metrics & Visualization Architecture

### 12.1 MetricsCollector

```python
@dataclass
class GenerationMetrics:
    """Per-generation snapshot of all tracked metrics."""
    generation: int
    population_size: int

    # Trait statistics
    trait_means: np.ndarray           # Mean of each trait
    trait_stds: np.ndarray            # Std dev of each trait
    trait_entropy: float              # Diversity measure

    # Region distribution
    region_counts: Dict[str, int]     # Count per processing region
    region_proportions: Dict[str, float]
    region_transitions: Dict[str, int]  # Transition counts this gen

    # Contribution
    total_contribution: float
    mean_contribution: float
    breakthrough_count: int

    # Suffering
    mean_suffering: float
    suffering_by_region: Dict[str, float]

    # Demographics
    births: int
    deaths: int
    mean_age: float
    pairs_formed: int
    dissolutions: int

    # Birth order analysis
    birth_order_trait_means: Dict[int, np.ndarray]  # Per birth order
    birth_order_contribution_means: Dict[int, float]

    # Outsider tracking
    outsider_descendant_count: int
    outsider_trait_diffusion: Optional[float]

    # Lore metrics
    active_memory_count: int
    myth_count: int
    mean_lore_fidelity: float

    # Extension metrics
    extension_metrics: Dict[str, Any]
```

### 12.2 Narrative Visualization System

Beyond raw data charts, the visualization system tells stories:

#### View 1: Dashboard / Mission Control
- Parameter panel with sliders for all config values
- Start/stop/step controls with generation counter
- Run comparison selector (overlay multiple runs)
- Preset selector and archetype founder picker
- Outsider injection controls (pick archetype, inject at current generation)

#### View 2: Population Overview (time series)
- **Trait drift heatmap**: N traits × generations, population mean with variance bands
- **Region population stacked area chart**: R1-R5 proportions over time
- **Population size** with birth/death overlay
- **Birth order composition** bars per generation
- **Breakthrough timeline**: scatter of events with agent details on hover

#### View 3: Agent Explorer
- Searchable/filterable agent list with trait sparklines
- **Individual agent detail**: trait radar chart, region timeline, contribution curve, family tree (parents → partner → children), cognitive council voice strengths, decision history
- **Agent comparison**: overlay two agents' trait radars
- **"Life of an Agent" timeline**: birth → trait changes → region transitions → pairing → children → breakthroughs → death, all on one horizontal band

#### View 4: Settlement Diagnostics (when geography/migration active)
- Per-location trait profile radar
- Settlement viability assessment with risk factors
- **"Why did this settlement fail?"** diagnostic panel
- Composition comparison across settlements
- Migration flow Sankey diagram

#### View 5: Experiment Comparison
- Side-by-side or overlaid time series across runs
- Statistical significance indicators for A/B tests
- Parameter diff table (what changed between runs)
- **Parameter sensitivity tornado chart**: which config params most affect outcomes

#### View 6: Family & Lineage
- **Dynasty tracker**: follow a lineage across all generations
- **Generational trait inheritance tree**: treemap or dendrogram showing trait flow through family lines
- Birth order effects visible in lineage coloring
- Outsider lineage highlighted in contrasting color

#### View 7: Network & Relationships
- **Attraction network graph**: nodes = agents (colored by processing region), edges = pairings (weight = attraction score)
- **Social bond network**: bond strengths between agents
- Cluster detection: emergent community groupings

#### View 8: Lore & Cultural Evolution (when lore/culture extensions active)
- Memory fidelity decay curves over generations
- Myth emergence timeline: when facts became myths
- Meme prevalence area chart
- Cultural influence on processing region distribution

#### View 9: Suffering vs. Contribution Analysis
- **Scatter plot**: per-agent suffering (x) vs. contribution (y), colored by region
- This is the key visualization for the R4/R5 distinction
- Interactive: click agent to see their trajectory over time
- Population-level trend lines per region

#### View 10: Anomaly Detection
- **"Generation X was special" detector**: automatically flag generations with unusual breakthrough rates, trait shifts, or region changes
- Explain what config/inheritance patterns produced the anomaly
- Historical comparison: "This generation looks like generation Y in run Z"

---

## Part 13: ExperimentConfig (Master Configuration)

```python
@dataclass
class ExperimentConfig:
    """
    Master configuration — ALL parameters as tunable sliders.
    Nothing in the simulation is hardcoded.
    """

    # === Experiment identity ===
    experiment_name: str = "default"
    random_seed: Optional[int] = None

    # === Trait system ===
    trait_preset: str = 'compact'          # 'compact' (15), 'full' (50), 'custom'
    custom_traits: Optional[List[dict]] = None  # For 'custom' preset

    # === Population ===
    initial_population: int = 100
    generations_to_run: int = 50
    founding_population: str = 'random'    # 'random', 'archetypes', 'custom'
    founding_archetypes: List[Tuple[str, int]] = field(default_factory=list)
    founding_trait_noise: float = 0.1

    # === Birth order rules ===
    birth_order_rules: Dict[int, str] = field(default_factory=lambda: {
        1: 'worst', 2: 'weirdest', 3: 'best'
    })  # 4+ defaults to 'random_weighted'
    inheritance_noise_sigma: float = 0.05

    # === RSH Processing regions ===
    region_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'under_to_optimal': 0.3,
        'optimal_to_deep': 0.5,
        'deep_to_extreme': 0.8,
        'productive_potential_threshold': 0.5,
    })
    productive_weights: Dict[str, float] = field(default_factory=lambda: {
        'creativity': 0.4,
        'resilience': 0.3,
        'burnout_penalty': 0.3,
    })

    # === Trait drift ===
    trait_drift_rate: float = 0.02
    trait_drift_age_factor: float = 0.01  # Drift slows with age
    region_effects: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # === Attraction model ===
    attraction_weights: Dict[str, float] = field(default_factory=lambda: {
        'similarity': 0.3,
        'complementarity': 0.2,
        'universal_attractiveness': 0.1,
        'social_proximity': 0.15,
        'age_compatibility': 0.1,
        'random_chemistry': 0.15,
    })

    # === Decision model ===
    decision_temperature: float = 1.0     # Higher = more random decisions
    decision_weights: Dict = field(default_factory=dict)   # Per-context W matrices
    decision_biases: Dict = field(default_factory=dict)    # Per-context bias vectors

    # === Cognitive council ===
    cognitive_council_enabled: bool = False  # Off by default for faster sims
    cognitive_council_weights: Dict = field(default_factory=dict)

    # === Relationships ===
    relationship_config: Dict[str, Any] = field(default_factory=lambda: {
        'pairing_min_age': 16,
        'pairing_permanent': False,
        'dissolution_enabled': True,
        'dissolution_compatibility_threshold': 0.3,
        'dissolution_base_rate': 0.05,
        'reparing_after_death': True,
        'reparing_after_dissolution': True,
        'reparing_cooldown_generations': 1,
        'infidelity_enabled': False,
        'infidelity_base_rate': 0.15,
        'single_by_choice_rate': 0.10,
        'lgbtq_rate': 0.035,
        'assisted_reproduction_rate': 0.075,
    })

    # === Fertility ===
    fertility_config: Dict[str, Any] = field(default_factory=lambda: {
        'female_fertility_start': 16,
        'female_fertility_end': 40,
        'male_fertility_start': 16,
        'min_birth_spacing_generations': 1,
        'max_children_per_generation': 1,
        'maternal_mortality_rate': 0.015,
        'child_mortality_rate': 0.30,
        'societal_fertility_pressure': 0.5,
        'target_children_mean': 3.0,
    })

    # === Mortality ===
    base_mortality_rate: float = 0.02
    age_mortality_factor: float = 0.001   # Increases with age
    burnout_mortality_factor: float = 0.1

    # === Lore/Memory system ===
    lore_enabled: bool = True
    lore_decay_rate: float = 0.05         # Fidelity loss per transmission
    lore_myth_threshold: float = 0.3      # Below this = myth
    lore_mutation_rate: float = 0.02      # Chance of content mutation
    lore_transmission_rate: float = 0.7   # Chance of passing to child

    # === Outsider injection ===
    outsider_injection_age: int = 20
    scheduled_injections: List[dict] = field(default_factory=list)
    # e.g., [{'generation': 25, 'archetype': 'einstein', 'count': 3}]

    # === Archetypes ===
    archetypes: Dict[str, np.ndarray] = field(default_factory=dict)
    # Loaded from archetype definitions

    # === Extensions ===
    extensions_enabled: List[str] = field(default_factory=list)
    extensions: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def enable_extension(self, extension: SimulationExtension):
        self.extensions_enabled.append(extension.name)
        self.extensions[extension.name] = extension.get_default_config()

    def configure_extension(self, name: str, **kwargs):
        if name in self.extensions:
            self.extensions[name].update(kwargs)

    def to_dict(self) -> dict:
        """Serialize to dict for saving/comparison."""
        pass

    @classmethod
    def from_dict(cls, d: dict) -> 'ExperimentConfig':
        """Deserialize from dict."""
        pass
```

---

## Part 14: Implementation Priority (Revised)

### Phase 1: Foundation (P0 — Core Engine)

| # | Component | Description |
|---|-----------|-------------|
| 1 | `TraitSystem` | Configurable trait definitions (compact/full/custom) |
| 2 | `ExperimentConfig` | Master config with all parameters |
| 3 | `Agent` | Core agent dataclass with history tracking |
| 4 | `ProcessingClassifier` | RSH Five Regions classification |
| 5 | `InheritanceEngine` | Birth order rules with configurable strategies |
| 6 | `TraitDriftEngine` | Experience and age-based trait drift |
| 7 | `AttractionModel` | Multi-factor attraction calculation |
| 8 | `DecisionModel` | Utility-based unified decision engine |
| 9 | `SimulationEngine` | Generation loop with all phases |

### Phase 2: Intelligence Layer (P1 — Decision & Memory)

| # | Component | Description |
|---|-----------|-------------|
| 10 | `CognitiveCouncil` | Optional sub-agent voting modulation |
| 11 | `LoreEngine` | Memory transmission and fidelity decay |
| 12 | `MetricsCollector` | Per-generation statistics collection |
| 13 | `ExperimentRunner` | A/B testing, parameter sweeps |
| 14 | `Presets` | Baseline, archetype-based, and experimental configs |
| 15 | `OutsiderInterface` + `RippleTracker` | Mid-sim injection and tracking |

### Phase 3: Visualization (P1 — See What's Happening)

| # | Component | Description |
|---|-----------|-------------|
| 16 | Dashboard / Mission Control | Parameter sliders, run controls |
| 17 | Population Overview | Time series, heatmaps, region charts |
| 18 | Agent Explorer | Individual agent deep-dive + comparison |
| 19 | Suffering vs. Contribution | The key R4/R5 visualization |
| 20 | Family & Lineage | Dynasty tracking, inheritance trees |

### Phase 4: Extensions (P2 — Add Complexity)

Revised order — migration elevated for agent orchestration relevance:

| # | Extension | Complexity | Dependencies |
|---|-----------|------------|--------------|
| 21 | Extension ABC + Registry | Low | None |
| 22 | Geography | Medium | None |
| 23 | Migration | High | Geography |
| 24 | Resources | Medium | None |
| 25 | Technology | Low | None |
| 26 | Culture/Memes | High | None |
| 27 | Conflict | Medium | None |

### Phase 5: Advanced Visualization (P2)

| # | Component | Description |
|---|-----------|-------------|
| 28 | Settlement Diagnostics | Composition analysis, failure forensics |
| 29 | Network Visualization | Attraction/social bond graphs |
| 30 | Lore Evolution View | Memory decay, myth emergence |
| 31 | Anomaly Detection | "Generation X was special" |
| 32 | Parameter Sensitivity | Tornado charts |

### Phase 6: LLM Integration (P3 — Build Last)

| # | Component | Description |
|---|-----------|-------------|
| 33 | Agent Interviewer | LLM-powered agent interviews |
| 34 | Narrative Generator | Natural language event summaries |
| 35 | Decision Narrator | Explain agent decisions in prose |

---

## Part 15: Key Experiments the System Should Support

These are the "what if?" questions the Seldon Sandbox exists to answer:

1. **Birth order hypothesis**: Does 1st=worst, 2nd=weirdest, 3rd=best produce emergent social structures? What happens with inverted rules?

2. **Optimal processing mix**: What proportion of R1/R2/R3/R4/R5 maximizes societal contribution? Is the ~5-10% sacrificial proportion from RSH a natural attractor?

3. **Settlement composition**: What personality mix makes a new settlement succeed? What mix causes failure?

4. **Outsider disruption**: How quickly do foreign traits propagate? Do they improve or destabilize communities?

5. **Cultural meme effects**: Does the "tortured genius" meme maintain the sacrificial processing population? What happens without it?

6. **Lore degradation**: How do distorted memories affect collective behavior? Can myths improve societal outcomes?

7. **Archetype societies**: What happens when you found a society entirely from Einsteins? From a mix of Curies and Fred Rogers?

8. **Resource scarcity pressure**: How does scarcity change processing region distributions? Does suffering increase or decrease innovation?

---

*Document Version: 3.0*
*Previous Version: 2.3*
*Key Changes: Configurable traits, utility-based decisions, cognitive council, lore system, outsider interface, narrative visualization, revised priorities*
*Core system is simple; complexity is opt-in through configuration and extensions*

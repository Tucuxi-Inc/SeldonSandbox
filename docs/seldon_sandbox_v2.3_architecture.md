# The Seldon Sandbox: A Psychohistory Engine
## Master Architecture Document v2.3

---

## Executive Summary

The Seldon Sandbox is an **experimentation platform** for multi-generational societal simulation. Every aspect of the system is exposed as a tunable parameter to explore "what if?" questions.

### Version 2.3 Changes

- **Extension Architecture** — Clean interfaces for future modules (geography, resources, migration, conflict)
- **Core vs. Extensions** — Start simple, expand later without refactoring
- **Module Stubs** — Placeholder interfaces showing where complexity can be added

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        EXPERIMENT RUNNER                             │
│                   (A/B testing, parameter sweeps)                    │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      EXPERIMENT CONFIG                               │
│            (All parameters as tunable sliders)                       │
└─────────────────────────────────────────────────────────────────────┘
                                  │
            ┌─────────────────────┼─────────────────────┐
            ▼                     ▼                     ▼
┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐
│    CORE ENGINE    │  │     EXTENSIONS    │  │     METRICS       │
│                   │  │    (Optional)     │  │                   │
│ • Agents (15-trait)│  │ • Geography       │  │ • Time series     │
│ • Inheritance     │  │ • Resources       │  │ • Visualization   │
│ • Processing      │  │ • Migration       │  │ • Comparison      │
│ • Attraction      │  │ • Conflict        │  │                   │
│ • Reproduction    │  │ • Culture/Memes   │  │                   │
│ • Mortality       │  │ • Technology      │  │                   │
└───────────────────┘  └───────────────────┘  └───────────────────┘
```

---

## Part 1: Core System (Build Now)

### What's in Core

These are the essential components we build first:

| Component | Description | Status |
|-----------|-------------|--------|
| `ExperimentConfig` | All tunable parameters | Build now |
| `Agent` | 15-trait model with drift | Build now |
| `TraitSystem` | Trait definitions and indices | Build now |
| `InheritanceEngine` | Configurable birth order rules | Build now |
| `ProcessingClassifier` | RSH Five Regions | Build now |
| `TraitDriftEngine` | Experience-based trait changes | Build now |
| `AttractionModel` | Pairing/mating selection | Build now |
| `SimulationLoop` | Generation processing | Build now |
| `MetricsCollector` | Time series tracking | Build now |

### Core Data Model

```python
@dataclass
class Agent:
    """Core agent - minimal but extensible."""
    id: str
    name: str
    age: int
    generation: int
    birth_order: int
    
    # Core traits (15-dimensional)
    traits: np.ndarray
    traits_at_birth: np.ndarray
    
    # Processing region
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
    
    # History (for visualization)
    trait_history: List[np.ndarray] = field(default_factory=list)
    region_history: List[ProcessingRegion] = field(default_factory=list)
    contribution_history: List[float] = field(default_factory=list)
    
    # === EXTENSION HOOKS ===
    # These are populated by extension modules if enabled
    location_id: Optional[str] = None          # Geography extension
    resource_holdings: Dict[str, float] = field(default_factory=dict)  # Resources extension
    cultural_memes: List[str] = field(default_factory=list)  # Culture extension
    skills: Dict[str, float] = field(default_factory=dict)  # Skills extension
    
    # Generic extension data (any module can store data here)
    extension_data: Dict[str, Any] = field(default_factory=dict)
```

---

## Part 2: Extension Architecture (Build Later)

### 2.1 Extension Interface

All extensions implement this interface:

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any


class SimulationExtension(ABC):
    """
    Base class for all simulation extensions.
    
    Extensions can:
    - Add parameters to ExperimentConfig
    - Hook into generation phases
    - Add metrics to track
    - Modify agent behavior
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique extension name."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """What this extension adds."""
        pass
    
    @abstractmethod
    def get_default_config(self) -> Dict[str, Any]:
        """Default configuration parameters for this extension."""
        pass
    
    def on_simulation_start(self, population: List[Agent], config: 'ExperimentConfig'):
        """Called once at simulation start."""
        pass
    
    def on_generation_start(self, generation: int, population: List[Agent], config: 'ExperimentConfig'):
        """Called at start of each generation."""
        pass
    
    def on_agent_created(self, agent: Agent, parents: tuple, config: 'ExperimentConfig'):
        """Called when a new agent is born."""
        pass
    
    def on_agent_decision(self, agent: Agent, decision_context: dict, config: 'ExperimentConfig') -> dict:
        """Modify or influence agent decisions. Return modified context."""
        return decision_context
    
    def on_generation_end(self, generation: int, population: List[Agent], config: 'ExperimentConfig'):
        """Called at end of each generation."""
        pass
    
    def get_metrics(self, population: List[Agent]) -> Dict[str, Any]:
        """Return extension-specific metrics to track."""
        return {}
    
    def modify_attraction(self, agent1: Agent, agent2: Agent, 
                          base_attraction: float, config: 'ExperimentConfig') -> float:
        """Modify attraction scores (e.g., based on proximity)."""
        return base_attraction
    
    def modify_mortality(self, agent: Agent, base_rate: float, 
                         config: 'ExperimentConfig') -> float:
        """Modify mortality rate (e.g., based on resources)."""
        return base_rate
```

### 2.2 Extension Registry

```python
class ExtensionRegistry:
    """
    Manages simulation extensions.
    """
    
    def __init__(self):
        self._extensions: Dict[str, SimulationExtension] = {}
        self._enabled: Dict[str, bool] = {}
    
    def register(self, extension: SimulationExtension):
        """Register an extension."""
        self._extensions[extension.name] = extension
        self._enabled[extension.name] = False
    
    def enable(self, name: str):
        """Enable an extension."""
        if name in self._extensions:
            self._enabled[name] = True
    
    def disable(self, name: str):
        """Disable an extension."""
        if name in self._extensions:
            self._enabled[name] = False
    
    def get_enabled(self) -> List[SimulationExtension]:
        """Get all enabled extensions."""
        return [ext for name, ext in self._extensions.items() 
                if self._enabled.get(name, False)]
    
    def get_combined_config(self) -> Dict[str, Any]:
        """Get combined default config from all enabled extensions."""
        combined = {}
        for ext in self.get_enabled():
            combined[ext.name] = ext.get_default_config()
        return combined
```

---

## Part 3: Planned Extensions (Stubs)

### 3.1 Geography Extension

```python
class GeographyExtension(SimulationExtension):
    """
    Adds spatial dimension to simulation.
    
    Features:
    - Hex grid or region-based map
    - Locations have carrying capacity
    - Distance affects attraction/interaction
    - Migration between locations
    """
    
    name = "geography"
    description = "Spatial simulation with locations and movement"
    
    def get_default_config(self) -> Dict[str, Any]:
        return {
            'enabled': False,
            'map_type': 'hex',  # 'hex', 'regions', 'continuous'
            'map_size': (10, 10),
            'starting_locations': 3,
            
            # Location parameters
            'base_carrying_capacity': 50,
            'capacity_variation': 0.3,
            
            # Distance effects
            'max_interaction_distance': 2,
            'attraction_distance_decay': 0.5,  # Attraction falls off with distance
            
            # Movement
            'migration_enabled': True,
            'migration_threshold': 0.3,  # How bad before people leave
        }
    
    def on_simulation_start(self, population, config):
        """Initialize map and place agents."""
        # Create locations
        # Assign initial positions
        pass
    
    def modify_attraction(self, agent1, agent2, base_attraction, config):
        """Reduce attraction for distant agents."""
        if not config.extensions.get('geography', {}).get('enabled'):
            return base_attraction
        
        distance = self._calculate_distance(agent1.location_id, agent2.location_id)
        decay = config.extensions['geography']['attraction_distance_decay']
        return base_attraction * (decay ** distance)


@dataclass
class Location:
    """A place agents can live."""
    id: str
    name: str
    coordinates: Tuple[int, int]
    
    # Capacity
    carrying_capacity: int = 50
    current_population: int = 0
    
    # Resources (if resource extension enabled)
    resource_richness: float = 1.0
    
    # Desirability
    base_desirability: float = 0.5
    
    # Connections to other locations
    connections: List[str] = field(default_factory=list)
    connection_difficulties: Dict[str, float] = field(default_factory=dict)
```

### 3.2 Resources Extension

```python
class ResourcesExtension(SimulationExtension):
    """
    Adds resource constraints.
    
    Features:
    - Agents need resources to survive
    - Resources are limited and regenerate
    - Resource scarcity affects mortality, fertility, conflict
    - Different resource types (food, shelter, status)
    """
    
    name = "resources"
    description = "Resource scarcity and economic dynamics"
    
    def get_default_config(self) -> Dict[str, Any]:
        return {
            'enabled': False,
            
            # Resource types
            'resource_types': ['food', 'shelter', 'status'],
            
            # Regeneration
            'base_regeneration_rate': 0.1,
            'carrying_capacity_resource_multiplier': 1.0,
            
            # Agent needs
            'consumption_per_agent': {
                'food': 1.0,
                'shelter': 0.5,
                'status': 0.0,  # Status is relative, not consumed
            },
            
            # Scarcity effects
            'scarcity_mortality_multiplier': 2.0,
            'scarcity_fertility_multiplier': 0.5,
            'scarcity_conflict_multiplier': 1.5,
            
            # Distribution
            'resource_distribution': 'equal',  # 'equal', 'trait_based', 'random'
            'hoarding_enabled': True,
        }
    
    def modify_mortality(self, agent, base_rate, config):
        """Increase mortality when resources scarce."""
        if not config.extensions.get('resources', {}).get('enabled'):
            return base_rate
        
        food = agent.resource_holdings.get('food', 0)
        need = config.extensions['resources']['consumption_per_agent']['food']
        
        if food < need:
            scarcity = (need - food) / need
            multiplier = config.extensions['resources']['scarcity_mortality_multiplier']
            return base_rate * (1 + scarcity * multiplier)
        
        return base_rate
```

### 3.3 Migration Extension

```python
class MigrationExtension(SimulationExtension):
    """
    Handles movement between locations.
    
    Features:
    - Push factors: overcrowding, scarcity, conflict
    - Pull factors: resources, family, opportunity
    - Migration difficulty based on distance, terrain
    - Group migration (founding new settlements)
    
    Key dynamics:
    - "Starting a new village is hard" - requires critical mass
    - "Wrong personality groupings can fail" - composition matters
    """
    
    name = "migration"
    description = "Population movement and settlement founding"
    
    def get_default_config(self) -> Dict[str, Any]:
        return {
            'enabled': False,
            'requires': ['geography'],  # Dependencies
            
            # Push factors (reasons to leave)
            'push_factors': {
                'overcrowding_weight': 0.3,
                'scarcity_weight': 0.3,
                'conflict_weight': 0.2,
                'low_status_weight': 0.2,
            },
            
            # Pull factors (reasons to go somewhere)
            'pull_factors': {
                'resources_weight': 0.3,
                'family_weight': 0.2,
                'opportunity_weight': 0.3,
                'safety_weight': 0.2,
            },
            
            # Migration mechanics
            'migration_decision_threshold': 0.6,  # Push-pull score needed to move
            'migration_cost': 0.2,  # Resource cost of moving
            
            # New settlement founding
            'new_settlement_enabled': True,
            'min_founding_group_size': 5,
            'founding_difficulty': 0.7,  # Base failure rate
            
            # Composition effects on settlement success
            'settlement_composition_effects': {
                'min_conscientiousness_mean': 0.4,  # Need organized people
                'max_neuroticism_mean': 0.7,  # Too much anxiety = failure
                'optimal_extraversion_variance': 0.2,  # Mix of social styles
                'requires_leader': True,  # Need someone with high dominance
            },
        }
    
    def evaluate_settlement_viability(self, founding_group: List[Agent], 
                                      config: 'ExperimentConfig') -> Tuple[float, List[str]]:
        """
        Evaluate whether a founding group can successfully start a new settlement.
        
        Returns (success_probability, risk_factors).
        
        This implements "wrong personality groupings can go bad."
        """
        comp_config = config.extensions['migration']['settlement_composition_effects']
        
        success_prob = 1.0 - config.extensions['migration']['founding_difficulty']
        risks = []
        
        # Get group trait statistics
        traits = np.array([a.traits for a in founding_group])
        means = traits.mean(axis=0)
        stds = traits.std(axis=0)
        
        # Check conscientiousness (organization needed)
        if means[TraitSystem.CONSCIENTIOUSNESS] < comp_config['min_conscientiousness_mean']:
            success_prob *= 0.5
            risks.append("Low organization/conscientiousness")
        
        # Check neuroticism (anxiety can doom a settlement)
        if means[TraitSystem.NEUROTICISM] > comp_config['max_neuroticism_mean']:
            success_prob *= 0.6
            risks.append("High collective anxiety")
        
        # Check for leadership
        if comp_config['requires_leader']:
            has_leader = any(a.traits[TraitSystem.DOMINANCE] > 0.7 for a in founding_group)
            if not has_leader:
                success_prob *= 0.4
                risks.append("No clear leader")
        
        # Check extraversion mix (all introverts or all extroverts is bad)
        extraversion_var = stds[TraitSystem.EXTRAVERSION]
        if abs(extraversion_var - comp_config['optimal_extraversion_variance']) > 0.2:
            success_prob *= 0.8
            risks.append("Imbalanced social composition")
        
        # Processing region diversity matters too
        regions = [a.processing_region for a in founding_group]
        if all(r == ProcessingRegion.SACRIFICIAL for r in regions):
            success_prob *= 0.3
            risks.append("All deep processors, no practical workers")
        if all(r == ProcessingRegion.UNDER_PROCESSING for r in regions):
            success_prob *= 0.7
            risks.append("No innovators in group")
        
        return (success_prob, risks)
```

### 3.4 Conflict Extension

```python
class ConflictExtension(SimulationExtension):
    """
    Handles interpersonal and inter-group conflict.
    
    Features:
    - Personality-based conflict triggers
    - Resource competition
    - Status competition
    - Inter-settlement conflict
    """
    
    name = "conflict"
    description = "Conflict dynamics and resolution"
    
    def get_default_config(self) -> Dict[str, Any]:
        return {
            'enabled': False,
            
            # Conflict triggers
            'triggers': {
                'resource_scarcity_threshold': 0.3,
                'dominance_clash_threshold': 0.8,  # Two high-dominance agents
                'trust_betrayal_threshold': 0.2,
            },
            
            # Personality effects on conflict
            'trait_conflict_weights': {
                'dominance': 0.3,      # High dominance = more conflict
                'agreeableness': -0.3,  # High agreeableness = less conflict
                'neuroticism': 0.2,    # High neuroticism = more conflict
            },
            
            # Resolution
            'resolution_methods': ['submission', 'compromise', 'separation', 'escalation'],
            'resolution_trait_influences': {
                'submission': {'dominance': -0.3, 'agreeableness': 0.2},
                'compromise': {'agreeableness': 0.3, 'openness': 0.2},
                'separation': {'extraversion': -0.2},
                'escalation': {'dominance': 0.3, 'neuroticism': 0.2},
            },
            
            # Consequences
            'conflict_suffering_cost': 0.2,
            'conflict_relationship_damage': 0.3,
        }
```

### 3.5 Culture/Memes Extension

```python
class CultureExtension(SimulationExtension):
    """
    Models cultural evolution and meme propagation.
    
    From RSH: Cultural memes like "tortured genius" maintain
    the ~5-10% sacrificial equilibrium.
    
    Features:
    - Memes spread through social networks
    - Memes affect behavior (fertility, risk-taking, processing)
    - Memes compete and evolve
    """
    
    name = "culture"
    description = "Cultural memes and their effects on behavior"
    
    def get_default_config(self) -> Dict[str, Any]:
        return {
            'enabled': False,
            
            # Starting memes
            'initial_memes': [
                {
                    'id': 'tortured_genius',
                    'description': 'Suffering is necessary for greatness',
                    'effects': {
                        'sacrificial_processing_bonus': 0.1,  # More likely to enter R4
                        'suffering_tolerance': 0.2,  # Tolerate more suffering
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
            
            # Meme dynamics
            'transmission_method': 'social_learning',  # 'social_learning', 'prestige_bias', 'conformity'
            'mutation_rate': 0.01,  # Memes can evolve
            'extinction_threshold': 0.05,  # Memes below this prevalence die out
        }
```

### 3.6 Technology Extension

```python
class TechnologyExtension(SimulationExtension):
    """
    Models technological progress and its effects.
    
    Features:
    - Tech level affects carrying capacity
    - Breakthroughs from R3/R4 agents advance tech
    - Tech can change the rules (mortality, fertility, etc.)
    """
    
    name = "technology"
    description = "Technological advancement and its societal effects"
    
    def get_default_config(self) -> Dict[str, Any]:
        return {
            'enabled': False,
            
            # Tech progression
            'starting_tech_level': 1.0,
            'breakthrough_tech_increment': 0.1,
            
            # Tech effects
            'tech_capacity_multiplier': 1.5,  # Per tech level
            'tech_mortality_reduction': 0.05,  # Per tech level
            'tech_fertility_bonus': 0.02,  # Per tech level
            
            # Tech loss
            'tech_decay_enabled': False,
            'tech_decay_rate': 0.01,  # Per generation without maintenance
            'tech_maintenance_population': 50,  # Min pop to maintain tech
        }
```

---

## Part 4: Integration Pattern

### 4.1 Generation Loop with Extensions

```python
class SimulationEngine:
    """
    Main simulation loop with extension hooks.
    """
    
    def __init__(self, config: ExperimentConfig, extensions: ExtensionRegistry):
        self.config = config
        self.extensions = extensions
        
        # Core components
        self.inheritance = InheritanceEngine(config)
        self.classifier = ProcessingClassifier(config)
        self.drift_engine = TraitDriftEngine(config)
        self.attraction = AttractionModel(config)
        self.metrics = MetricsCollector(config)
    
    def run(self, generations: int) -> MetricsCollector:
        """Run the simulation."""
        population = self._create_initial_population()
        
        # Extension: simulation start
        for ext in self.extensions.get_enabled():
            ext.on_simulation_start(population, self.config)
        
        for gen in range(generations):
            events = self._run_generation(gen, population)
            self.metrics.collect(gen, population, events)
        
        return self.metrics
    
    def _run_generation(self, generation: int, population: List[Agent]) -> dict:
        """Run one generation with extension hooks."""
        events = {'births': 0, 'deaths': 0, 'breakthroughs': 0}
        
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
            new_region = self.classifier.process_region_transitions(agent)
            if new_region != agent.processing_region:
                events.setdefault('transitions', {})[f"{agent.processing_region.value}_to_{new_region.value}"] = \
                    events.get('transitions', {}).get(f"{agent.processing_region.value}_to_{new_region.value}", 0) + 1
            agent.processing_region = new_region
            agent.region_history.append(agent.processing_region)
        
        # === Phase 3: Contribution and breakthroughs ===
        for agent in population:
            contribution, breakthrough = self._calculate_contribution(agent)
            agent.contribution_history.append(contribution)
            if breakthrough:
                events['breakthroughs'] += 1
        
        # === Phase 4: Pairing ===
        eligible = [a for a in population if a.age >= 16 and a.partner_id is None and a.is_alive]
        pairs = self._form_pairs(eligible)
        
        # === Phase 5: Reproduction ===
        new_children = []
        for p1, p2 in pairs:
            if self._will_reproduce(p1, p2):
                child = self._create_child(p1, p2, population)
                new_children.append(child)
                events['births'] += 1
                
                # === Extension hook: agent created ===
                for ext in self.extensions.get_enabled():
                    ext.on_agent_created(child, (p1, p2), self.config)
        
        population.extend(new_children)
        
        # === Phase 6: Mortality ===
        for agent in population:
            base_rate = self._calculate_base_mortality(agent)
            
            # === Extension hook: modify mortality ===
            for ext in self.extensions.get_enabled():
                base_rate = ext.modify_mortality(agent, base_rate, self.config)
            
            if np.random.random() < base_rate:
                agent.is_alive = False
                events['deaths'] += 1
        
        population[:] = [a for a in population if a.is_alive]
        
        # === Extension hook: generation end ===
        for ext in self.extensions.get_enabled():
            ext.on_generation_end(generation, population, self.config)
        
        # === Collect extension metrics ===
        for ext in self.extensions.get_enabled():
            ext_metrics = ext.get_metrics(population)
            events[f'ext_{ext.name}'] = ext_metrics
        
        return events
    
    def _form_pairs(self, eligible: List[Agent]) -> List[Tuple[Agent, Agent]]:
        """Form pairs with extension-modified attraction."""
        pairs = []
        unpaired = eligible.copy()
        
        while len(unpaired) >= 2:
            a1 = unpaired.pop(0)
            
            # Calculate attraction to all others
            attractions = []
            for a2 in unpaired:
                base = self.attraction.calculate(a1, a2)
                
                # === Extension hook: modify attraction ===
                for ext in self.extensions.get_enabled():
                    base = ext.modify_attraction(a1, a2, base, self.config)
                
                attractions.append((a2, base))
            
            # Select partner (weighted by attraction)
            if attractions:
                agents, scores = zip(*attractions)
                scores = np.array(scores)
                scores = np.maximum(scores, 0)  # No negative
                if scores.sum() > 0:
                    probs = scores / scores.sum()
                    idx = np.random.choice(len(agents), p=probs)
                    partner = agents[idx]
                    unpaired.remove(partner)
                    pairs.append((a1, partner))
        
        return pairs
```

---

## Part 5: What to Build Now vs Later

### Build Now (Core)

| Component | Priority | Est. Time |
|-----------|----------|-----------|
| `ExperimentConfig` | P0 | 2 days |
| `Agent` + `TraitSystem` | P0 | 1 day |
| `InheritanceEngine` | P0 | 2 days |
| `ProcessingClassifier` | P0 | 1 day |
| `TraitDriftEngine` | P0 | 1 day |
| `AttractionModel` | P0 | 1 day |
| `SimulationEngine` (basic loop) | P0 | 2 days |
| `MetricsCollector` | P0 | 2 days |
| `ExperimentRunner` | P1 | 2 days |
| Basic visualization | P1 | 3 days |

**Total Core: ~4-5 weeks**

### Build Later (Extensions)

| Extension | Complexity | Dependencies |
|-----------|------------|--------------|
| Geography | Medium | None |
| Resources | Medium | None |
| Migration | High | Geography |
| Conflict | Medium | None |
| Culture/Memes | High | None |
| Technology | Low | None |

**Each extension: 1-2 weeks**

### Recommended Order

1. **Core system** — Get basic simulation running
2. **Visualization** — See what's happening
3. **Technology** — Simplest extension, adds breakthrough effects
4. **Geography** — Adds space without too much complexity  
5. **Resources** — Adds scarcity dynamics
6. **Migration** — Builds on geography + resources
7. **Culture** — Most complex, adds meme dynamics
8. **Conflict** — Can wait, adds drama but not essential

---

## Part 6: Extension Configuration in ExperimentConfig

```python
@dataclass
class ExperimentConfig:
    """
    Master configuration with extension support.
    """
    
    # ... all existing core parameters ...
    
    # === EXTENSION CONFIGURATION ===
    extensions_enabled: List[str] = field(default_factory=list)
    
    # Extension-specific configs (populated when extensions enabled)
    extensions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def enable_extension(self, extension: SimulationExtension):
        """Enable an extension and add its default config."""
        self.extensions_enabled.append(extension.name)
        self.extensions[extension.name] = extension.get_default_config()
    
    def configure_extension(self, name: str, **kwargs):
        """Update extension configuration."""
        if name in self.extensions:
            self.extensions[name].update(kwargs)


# Example usage
config = ExperimentConfig()
config.enable_extension(GeographyExtension())
config.configure_extension('geography', map_size=(20, 20), migration_enabled=True)
```

---

*Document Version: 2.3*
*Key Change: Clean extension architecture for future modules*
*Core system is simple; complexity is opt-in*

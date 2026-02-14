export interface SessionSummary {
  id: string;
  name: string;
  status: string;
  current_generation: number;
  max_generations: number;
  population_size: number;
}

export interface SessionResponse extends SessionSummary {
  config: Record<string, unknown>;
}

export interface AgentSummary {
  id: string;
  name: string;
  age: number;
  generation: number;
  birth_order: number;
  processing_region: string;
  suffering: number;
  is_alive: boolean;
  partner_id: string | null;
  is_outsider: boolean;
  dominant_voice: string | null;
  latest_contribution: number;
  has_decisions: boolean;
  location: number[] | null;
}

export interface AgentDetail extends AgentSummary {
  traits: Record<string, number>;
  traits_at_birth: Record<string, number>;
  trait_history: Record<string, number>[];
  region_history: string[];
  contribution_history: number[];
  suffering_history: number[];
  parent1_id: string | null;
  parent2_id: string | null;
  children_ids: string[];
  relationship_status: string;
  burnout_level: number;
  personal_memories: Record<string, unknown>[];
  inherited_lore: Record<string, unknown>[];
  decision_history: Record<string, unknown>[];
  outsider_origin: string | null;
  injection_generation: number | null;
}

export interface PaginatedAgentList {
  agents: AgentSummary[];
  total: number;
  page: number;
  page_size: number;
}

export interface GenerationMetrics {
  generation: number;
  population_size: number;
  births: number;
  deaths: number;
  breakthroughs: number;
  pairs_formed: number;
  trait_means: Record<string, number>;
  trait_stds: Record<string, number>;
  trait_entropy: number;
  region_counts: Record<string, number>;
  region_fractions: Record<string, number>;
  region_transitions: Record<string, number>;
  total_contribution: number;
  mean_contribution: number;
  max_contribution: number;
  mean_suffering: number;
  suffering_by_region: Record<string, number>;
  mean_age: number;
  age_distribution: Record<string, number>;
  birth_order_counts: Record<string, number>;
  total_memories: number;
  societal_memories: number;
  myths_count: number;
  outsider_count: number;
  outsider_descendant_count: number;
  dissolutions: number;
  infidelity_events: number;
  outsiders_injected: number;
  dominant_voice_counts: Record<string, number>;
}

export interface SummaryStats {
  total_generations: number;
  final_population_size: number;
  total_breakthroughs: number;
  mean_contribution: number;
  mean_suffering: number;
  peak_population: number;
  total_births: number;
  total_deaths: number;
}

export interface TimeSeries {
  field: string;
  generations: number[];
  values: unknown[];
}

export interface PresetInfo {
  name: string;
  config: Record<string, unknown>;
}

export interface ArchetypeInfo {
  name: string;
  display_name: string;
  description: string;
  key_traits: string[];
  use_case: string;
}

export interface ArchetypeDetail extends ArchetypeInfo {
  trait_values: Record<string, number>;
}

export interface FamilyTreeNode {
  id: string;
  name: string;
  generation: number;
  birth_order: number;
  processing_region: string;
  is_alive: boolean;
  is_outsider: boolean;
  child_id?: string;
  parent_id?: string;
  children?: FamilyTreeNode[];
}

export interface FamilyTree {
  root: FamilyTreeNode | null;
  ancestors: FamilyTreeNode[];
  descendants: FamilyTreeNode[];
}

export interface ComparisonResponse {
  sessions: Record<string, SummaryStats>;
  config_diffs: Record<string, Record<string, unknown[]>>;
}

// === Anomaly Detection ===

export interface Anomaly {
  generation: number;
  severity: string;
  z_score: number;
  metric: string;
  value: number;
  mean: number;
  std: number;
  category: string;
  description: string;
}

export interface AnomalyReport {
  anomalies: Anomaly[];
  generation_scores: number[];
  thresholds: { warning: number; anomaly: number; critical: number };
}

// === Lore Evolution ===

export interface LoreMemory {
  id: string;
  content: string;
  memory_type: string;
  fidelity: number;
  emotional_valence: number;
  trait_modifiers: Record<string, number>;
  created_generation: number;
  source_agent_id: string | null;
  transmission_count: number;
  mutation_count: number;
}

export interface LoreOverview {
  time_series: {
    total_memories: number[];
    societal_memories: number[];
    myths_count: number[];
    generations: number[];
    mean_fidelity?: number[];
  };
  current_societal_lore: LoreMemory[];
  memory_type_distribution: Record<string, number>;
}

export interface MemePrevalence {
  enabled: boolean;
  memes?: { id: string; name: string; prevalence: number; effects: Record<string, number> }[];
  prevalence_over_time: Record<string, number[]>;
  generations?: number[];
  current_dominant?: string | null;
}

// === Settlements ===

export interface Settlement {
  id: string;
  name: string;
  coordinates: number[];
  population: number;
  carrying_capacity: number;
  occupancy_ratio: number;
  resource_richness: number;
  region_counts: Record<string, number>;
}

export interface SettlementOverview {
  enabled: boolean;
  settlements: Settlement[];
  total_capacity: number;
  total_population: number;
}

export interface ViabilityAssessment {
  location_id: string;
  viability_score: number;
  risk_factors: string[];
  checks_passed: number;
  checks_total: number;
  group_size: number;
}

export interface MigrationEvent {
  generation: number;
  type: string;
  location_id?: string;
  founders?: number;
  count?: number;
  viability?: number;
  risks?: string[];
  from?: string;
  to?: string;
}

export interface MigrationHistory {
  enabled: boolean;
  events: MigrationEvent[];
  timeline: {
    settlement_count_by_gen: number[];
    migrations_by_gen: number[];
  };
}

// === Network ===

export interface NetworkNode {
  id: string;
  name: string;
  region: string;
  location_id: string | null;
}

export interface NetworkEdge {
  source: string;
  target: string;
  type: string;
  strength: number;
}

export interface NetworkGraph {
  nodes: NetworkNode[];
  edges: NetworkEdge[];
  stats: {
    total_nodes: number;
    total_edges: number;
    avg_connections: number;
    connected_components: number;
  };
}

// === Sensitivity ===

export interface ParameterSensitivity {
  parameter: string;
  correlation: number;
  impact: number;
  min_value: number | string;
  max_value: number | string;
  min_outcome: number;
  max_outcome: number;
}

export interface TornadoBar {
  parameter: string;
  low_outcome: number;
  high_outcome: number;
  low_value: number | string;
  high_value: number | string;
  swing: number;
}

export interface SensitivityReport {
  target_metric: string;
  session_count: number;
  sensitivities: ParameterSensitivity[];
  tornado_data: TornadoBar[];
}

// === LLM ===

export interface LLMProviderInfo {
  available: boolean;
  has_key?: boolean;
  models?: string[];
  base_url?: string;
}

export interface LLMStatus {
  available: boolean;
  message: string;
  providers: {
    anthropic: LLMProviderInfo;
    ollama: LLMProviderInfo;
  };
}

export interface InterviewResponse {
  response: string;
  model: string;
  input_tokens: number;
  output_tokens: number;
  provider: string;
}

export interface NarrativeResponse {
  narrative: string;
  generation: number;
  model: string;
  input_tokens: number;
  output_tokens: number;
  provider: string;
}

export interface DecisionExplainResponse {
  explanation: string;
  decision: Record<string, unknown>;
  model: string;
  input_tokens: number;
  output_tokens: number;
  provider: string;
}

// === Outsider Tracking ===

export interface OutsiderImpact {
  agent: AgentSummary;
  descendant_count: number;
  descendants: AgentSummary[];
  trait_distance_from_mean: number;
  injection_generation: number | null;
  outsider_origin: string | null;
  gender: string | null;
}

// === Community / Diplomacy ===

export interface CommunityProfile {
  id: string;
  name: string;
  population: number;
  trait_means: Record<string, number>;
  cohesion: number;
  dominant_region: string;
}

export interface DiplomaticRelation {
  community_a: string;
  community_b: string;
  standing: number;
  status: string;
}

export interface CommunityOverview {
  enabled: boolean;
  communities: CommunityProfile[];
  diplomatic_relations?: DiplomaticRelation[];
}

// === Economics ===

export interface EconomicsOverview {
  enabled: boolean;
  gdp_by_community: Record<string, number>;
  gini_coefficient: number;
  total_wealth: number;
}

export interface TradeRoute {
  from_community: string;
  to_community: string;
  volume: number;
  resource: string;
}

export interface WealthDistribution {
  enabled: boolean;
  percentiles: Record<string, number>;
  mean_wealth: number;
  median_wealth: number;
}

export interface OccupationBreakdown {
  enabled: boolean;
  occupations: Record<string, number>;
}

// === Environment ===

export interface ClimateState {
  enabled: boolean;
  current_season: string;
  season_progress: number;
  locations: Record<string, { temperature: number; rainfall: number }>;
}

export interface EnvironmentEvent {
  generation: number;
  type: string;
  severity: string;
  location_id: string | null;
  description: string;
}

export interface DiseaseInfo {
  name: string;
  infection_count: number;
  mortality_rate: number;
  active: boolean;
}

// === Hierarchy ===

export interface HierarchyOverview {
  enabled: boolean;
  status_distribution: Record<string, number>;
  mean_status: number;
}

export interface RoleBreakdown {
  enabled: boolean;
  roles: Record<string, number>;
}

export interface InfluenceEntry {
  agent_id: string;
  agent_name: string;
  influence_score: number;
  social_role: string | null;
}

export interface MentorshipChain {
  mentor_id: string;
  mentor_name: string;
  mentees: { id: string; name: string }[];
}

// === Genetics ===

export interface AlleleFrequencies {
  enabled: boolean;
  loci: Record<string, {
    trait: string;
    dominant_count: number;
    recessive_count: number;
    dominant_frequency: number;
    recessive_frequency: number;
  }>;
}

export interface EpigeneticPrevalence {
  enabled: boolean;
  markers: Record<string, {
    active_count: number;
    total: number;
    prevalence: number;
  }>;
}

export interface TraitGeneCorrelation {
  enabled: boolean;
  correlations: Record<string, number>;
}

// === Beliefs / Epistemology ===

export interface BeliefOverview {
  enabled: boolean;
  total_beliefs: number;
  mean_conviction: number;
  mean_accuracy: number;
  epistemology_distribution: Record<string, number>;
  domain_distribution: Record<string, number>;
  societal_beliefs: Record<string, unknown>[];
}

export interface EpistemologyDistribution {
  enabled: boolean;
  distribution: Record<string, { count: number; mean_accuracy: number }>;
}

export interface AccuracyByDomain {
  enabled: boolean;
  domains: Record<string, { count: number; mean_accuracy: number }>;
}

// === Inner Life / Experiential Mind ===

export interface InnerLifeOverview {
  enabled: boolean;
  mean_phenomenal_quality: number;
  pq_distribution: Record<string, number>;
  mean_experience_count: number;
  event_type_counts: Record<string, number>;
  population_mood: number[];
  mean_drift_magnitude: number;
}

export interface InnerLifeAgentState {
  enabled: boolean;
  agent_id: string;
  state: {
    experiences: Record<string, unknown>[];
    phenomenal_quality: number;
    pq_history: number[];
    mood: number[];
    experiential_drift_applied: Record<string, number>;
  } | null;
}

export interface PQDistribution {
  enabled: boolean;
  distribution: Record<string, number>;
  stats: { mean: number; std: number; min: number; max: number; count: number };
}

export interface MoodMapAgent {
  agent_id: string;
  mood: Record<string, number>;
  phenomenal_quality: number;
}

export interface MoodMapResponse {
  enabled: boolean;
  agents: MoodMapAgent[];
}

export interface ExperientialDriftResponse {
  enabled: boolean;
  drift_by_trait: Record<string, { mean_drift: number; max_drift: number; agents_affected: number }>;
  agents_with_drift: number;
}

// === Hex Grid ===

export interface HexTileData {
  q: number;
  r: number;
  terrain_type: string;
  elevation: number;
  water_access: number;
  soil_quality: number;
  natural_resources: number;
  vegetation: number;
  wildlife: number;
  habitability: number;
  capacity: number;
  agent_count: number;
  agent_ids: string[];
  region_counts: Record<string, number>;
  agents: { id: string; name: string; processing_region: string }[];
}

export interface HexGridCluster {
  tiles: number[][];
  agent_count: number;
  agent_ids: string[];
  center: number[];
  terrain_types: string[];
}

export interface HexGridResponse {
  enabled: boolean;
  tiles?: HexTileData[];
  clusters?: HexGridCluster[];
  stats?: {
    total_tiles: number;
    habitable_tiles: number;
    occupied_tiles: number;
    total_agents_on_grid: number;
    width: number;
    height: number;
  };
}

export interface HexTileDetailResponse {
  enabled: boolean;
  tile?: {
    q: number;
    r: number;
    terrain_type: string;
    elevation: number;
    water_access: number;
    soil_quality: number;
    natural_resources: number;
    vegetation: number;
    wildlife: number;
    habitability: number;
    capacity: number;
    agent_count: number;
    is_habitable: boolean;
  };
  agents?: AgentSummary[];
  neighbors?: {
    q: number;
    r: number;
    terrain_type: string;
    agent_count: number;
    habitability: number;
  }[];
}

// === Chronicle / Biography ===

export interface NotableEvent {
  event_type: string;
  generation: number;
  severity: string;
  headline: string;
  detail: string;
  agent_ids: string[];
  metrics_snapshot: Record<string, unknown>;
}

export interface DeathAnalysis {
  generation: number;
  age_at_death: number;
  mortality_breakdown: Record<string, number>;
  primary_cause: string;
  total_mortality_rate: number;
  processing_region_at_death: string;
  suffering_at_death: number;
  burnout_at_death: number;
}

export interface AgentBiography {
  agent: AgentSummary;
  identity: {
    name: string;
    age: number;
    generation: number;
    birth_order: number;
    is_outsider: boolean;
    outsider_origin: string | null;
    gender: string | null;
  };
  personality_profile: {
    top_traits: { name: string; value: number }[];
    processing_region: string;
    dominant_voice: string | null;
    region_journey: string[];
  };
  life_timeline: NotableEvent[];
  relationships: {
    partner: { id: string; name: string } | null;
    parents: { id: string; name: string }[];
    children: { id: string; name: string }[];
  };
  contribution_summary: {
    peak: number;
    mean: number;
    total_generations: number;
    has_breakthrough: boolean;
  };
  death_analysis: DeathAnalysis | null;
  prose: string | null;
  prose_error?: string | null;
}

export interface ChronicleEntry {
  generation: number;
  events: NotableEvent[];
  agent_names?: Record<string, string>;
  population_size: number;
  births: number;
  deaths: number;
}

export interface ChronicleIndex {
  generations: {
    generation: number;
    event_count: number;
    max_severity: string;
    population_size: number;
  }[];
}

export interface AgentComparisonResult {
  agents: AgentDetail[];
  relationships: {
    type: string;
    agent_a: string;
    agent_b: string;
    detail: string;
  }[];
}

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

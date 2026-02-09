import axios from 'axios';
import type {
  SessionResponse,
  SessionSummary,
  GenerationMetrics,
  SummaryStats,
  TimeSeries,
  PaginatedAgentList,
  AgentDetail,
  FamilyTree,
  PresetInfo,
  ArchetypeInfo,
  ArchetypeDetail,
  ComparisonResponse,
  AgentSummary,
  AnomalyReport,
  LoreOverview,
  MemePrevalence,
  SettlementOverview,
  ViabilityAssessment,
  MigrationHistory,
  NetworkGraph,
  SensitivityReport,
  LLMStatus,
  InterviewResponse,
  NarrativeResponse,
  DecisionExplainResponse,
  OutsiderImpact,
  CommunityOverview,
  EconomicsOverview,
  TradeRoute,
  WealthDistribution,
  OccupationBreakdown,
  ClimateState,
  EnvironmentEvent,
  DiseaseInfo,
  HierarchyOverview,
  RoleBreakdown,
  InfluenceEntry,
  MentorshipChain,
  AlleleFrequencies,
  EpigeneticPrevalence,
  TraitGeneCorrelation,
} from '../types';

const api = axios.create({ baseURL: '/api' });

// === Simulation ===

export async function createSession(params: {
  config?: Record<string, unknown>;
  preset?: string;
  name?: string;
}): Promise<SessionResponse> {
  const { data } = await api.post('/simulation/sessions', params);
  return data;
}

export async function listSessions(): Promise<SessionSummary[]> {
  const { data } = await api.get('/simulation/sessions');
  return data;
}

export async function getSession(id: string): Promise<SessionResponse> {
  const { data } = await api.get(`/simulation/sessions/${id}`);
  return data;
}

export async function deleteSession(id: string): Promise<void> {
  await api.delete(`/simulation/sessions/${id}`);
}

export async function runSession(id: string, generations?: number): Promise<SessionResponse> {
  const { data } = await api.post(`/simulation/sessions/${id}/run`, { generations });
  return data;
}

export async function stepSession(id: string, n: number = 1): Promise<SessionResponse> {
  const { data } = await api.post(`/simulation/sessions/${id}/step`, { n });
  return data;
}

export async function resetSession(id: string): Promise<SessionResponse> {
  const { data } = await api.post(`/simulation/sessions/${id}/reset`);
  return data;
}

// === Metrics ===

export async function getGenerations(
  sessionId: string,
  fromGen?: number,
  toGen?: number,
): Promise<GenerationMetrics[]> {
  const params: Record<string, number> = {};
  if (fromGen !== undefined) params.from_gen = fromGen;
  if (toGen !== undefined) params.to_gen = toGen;
  const { data } = await api.get(`/metrics/${sessionId}/generations`, { params });
  return data;
}

export async function getTimeSeries(sessionId: string, field: string): Promise<TimeSeries> {
  const { data } = await api.get(`/metrics/${sessionId}/time-series/${field}`);
  return data;
}

export async function getSummary(sessionId: string): Promise<SummaryStats> {
  const { data } = await api.get(`/metrics/${sessionId}/summary`);
  return data;
}

// === Agents ===

export async function listAgents(
  sessionId: string,
  params: {
    alive_only?: boolean;
    region?: string;
    generation?: number;
    birth_order?: number;
    is_outsider?: boolean;
    search?: string;
    page?: number;
    page_size?: number;
  } = {},
): Promise<PaginatedAgentList> {
  const { data } = await api.get(`/agents/${sessionId}`, { params });
  return data;
}

export async function getAgentDetail(sessionId: string, agentId: string): Promise<AgentDetail> {
  const { data } = await api.get(`/agents/${sessionId}/${agentId}`);
  return data;
}

export async function getFamilyTree(
  sessionId: string,
  agentId: string,
  depthUp: number = 3,
  depthDown: number = 3,
): Promise<FamilyTree> {
  const { data } = await api.get(`/agents/${sessionId}/${agentId}/family-tree`, {
    params: { depth_up: depthUp, depth_down: depthDown },
  });
  return data;
}

// === Experiments ===

export async function getPresets(): Promise<PresetInfo[]> {
  const { data } = await api.get('/experiments/presets');
  return data;
}

export async function getArchetypes(): Promise<ArchetypeInfo[]> {
  const { data } = await api.get('/experiments/archetypes');
  return data;
}

export async function getArchetypeDetail(name: string): Promise<ArchetypeDetail> {
  const { data } = await api.get(`/experiments/archetypes/${name}`);
  return data;
}

export async function compareSessions(sessionIds: string[]): Promise<ComparisonResponse> {
  const { data } = await api.post('/experiments/compare', { session_ids: sessionIds });
  return data;
}

export async function injectOutsider(params: {
  session_id: string;
  archetype?: string;
  custom_traits?: Record<string, number>;
  noise_sigma?: number;
  name?: string;
  gender?: string;
  age?: number;
  injection_generation?: number;
}): Promise<AgentSummary> {
  const { data } = await api.post('/experiments/inject-outsider', params);
  return data;
}

export async function getRippleReport(sessionId: string): Promise<Record<string, unknown>> {
  const { data } = await api.get(`/experiments/${sessionId}/ripple`);
  return data;
}

// === Advanced (Anomaly, Lore, Sensitivity) ===

export async function getAnomalies(sessionId: string): Promise<AnomalyReport> {
  const { data } = await api.get(`/advanced/${sessionId}/anomalies`);
  return data;
}

export async function getLoreOverview(sessionId: string): Promise<LoreOverview> {
  const { data } = await api.get(`/advanced/${sessionId}/lore/overview`);
  return data;
}

export async function getMemePrevalence(sessionId: string): Promise<MemePrevalence> {
  const { data } = await api.get(`/advanced/${sessionId}/lore/meme-prevalence`);
  return data;
}

export async function computeSensitivity(
  sessionId: string,
  sessionIds: string[],
  targetMetric: string,
): Promise<SensitivityReport> {
  const { data } = await api.post(`/advanced/${sessionId}/sensitivity`, {
    session_ids: sessionIds,
    target_metric: targetMetric,
  });
  return data;
}

// === Settlements ===

export async function getSettlementsOverview(sessionId: string): Promise<SettlementOverview> {
  const { data } = await api.get(`/settlements/${sessionId}/overview`);
  return data;
}

export async function getSettlementViability(
  sessionId: string,
  locationId: string,
): Promise<ViabilityAssessment> {
  const { data } = await api.get(`/settlements/${sessionId}/viability/${locationId}`);
  return data;
}

export async function getMigrationHistory(sessionId: string): Promise<MigrationHistory> {
  const { data } = await api.get(`/settlements/${sessionId}/migration-history`);
  return data;
}

export async function getSettlementComposition(
  sessionId: string,
  locationId: string,
): Promise<Record<string, unknown>> {
  const { data } = await api.get(`/settlements/${sessionId}/settlement-composition/${locationId}`);
  return data;
}

// === Network ===

export async function getNetworkGraph(
  sessionId: string,
  bondThreshold: number = 0.1,
): Promise<NetworkGraph> {
  const { data } = await api.get(`/network/${sessionId}/graph`, {
    params: { bond_threshold: bondThreshold },
  });
  return data;
}

// === LLM ===

export async function getLLMStatus(): Promise<LLMStatus> {
  const { data } = await api.get('/llm/status');
  return data;
}

export async function setLLMApiKey(apiKey: string): Promise<{ status: string; message: string }> {
  const { data } = await api.post('/llm/api-key', { api_key: apiKey });
  return data;
}

export async function clearLLMApiKey(): Promise<{ status: string; message: string }> {
  const { data } = await api.delete('/llm/api-key');
  return data;
}

export async function setOllamaUrl(baseUrl: string): Promise<{ status: string; message: string; base_url?: string }> {
  const { data } = await api.post('/llm/ollama-url', { base_url: baseUrl });
  return data;
}

export async function testLLMConnection(params: {
  provider: string;
  model?: string;
  api_key?: string;
  ollama_base_url?: string;
}): Promise<{ success: boolean; message: string; models?: string[]; base_url?: string }> {
  const { data } = await api.post('/llm/test-connection', params);
  return data;
}

export async function interviewAgent(
  sessionId: string,
  agentId: string,
  question: string,
  conversationHistory?: { role: string; content: string }[],
  provider: string = 'anthropic',
  model?: string,
): Promise<InterviewResponse> {
  const { data } = await api.post(`/llm/${sessionId}/interview/${agentId}`, {
    question,
    conversation_history: conversationHistory,
    provider,
    model: model || undefined,
  });
  return data;
}

export async function getGenerationNarrative(
  sessionId: string,
  generation: number,
  provider: string = 'anthropic',
  model?: string,
): Promise<NarrativeResponse> {
  const { data } = await api.get(`/llm/${sessionId}/narrative/${generation}`, {
    params: { provider, model: model || undefined },
  });
  return data;
}

export async function explainDecision(
  sessionId: string,
  agentId: string,
  decisionIndex: number,
  provider: string = 'anthropic',
  model?: string,
): Promise<DecisionExplainResponse> {
  const { data } = await api.post(`/llm/${sessionId}/decision-explain`, {
    agent_id: agentId,
    decision_index: decisionIndex,
    provider,
    model: model || undefined,
  });
  return data;
}

// === Experiments: Trait Names & Outsiders ===

export async function getTraitNames(): Promise<string[]> {
  const { data } = await api.get('/experiments/trait-names');
  return data;
}

export async function getOutsiders(sessionId: string): Promise<AgentSummary[]> {
  const { data } = await api.get(`/experiments/${sessionId}/outsiders`);
  return data;
}

export async function getOutsiderImpact(sessionId: string, agentId: string): Promise<OutsiderImpact> {
  const { data } = await api.get(`/experiments/${sessionId}/outsiders/${agentId}/impact`);
  return data;
}

// === Communities ===

export async function getCommunities(sessionId: string): Promise<CommunityOverview> {
  const { data } = await api.get(`/communities/${sessionId}/communities`);
  return data;
}

export async function getDiplomacy(sessionId: string): Promise<Record<string, unknown>> {
  const { data } = await api.get(`/communities/${sessionId}/diplomacy`);
  return data;
}

// === Economics ===

export async function getEconomicsOverview(sessionId: string): Promise<EconomicsOverview> {
  const { data } = await api.get(`/economics/${sessionId}/overview`);
  return data;
}

export async function getTradeRoutes(sessionId: string): Promise<{ enabled: boolean; routes: TradeRoute[] }> {
  const { data } = await api.get(`/economics/${sessionId}/trade-routes`);
  return data;
}

export async function getWealthDistribution(sessionId: string): Promise<WealthDistribution> {
  const { data } = await api.get(`/economics/${sessionId}/wealth-distribution`);
  return data;
}

export async function getOccupations(sessionId: string): Promise<OccupationBreakdown> {
  const { data } = await api.get(`/economics/${sessionId}/occupations`);
  return data;
}

// === Environment ===

export async function getClimate(sessionId: string): Promise<ClimateState> {
  const { data } = await api.get(`/environment/${sessionId}/climate`);
  return data;
}

export async function getEnvironmentEvents(sessionId: string): Promise<{ enabled: boolean; events: EnvironmentEvent[] }> {
  const { data } = await api.get(`/environment/${sessionId}/events`);
  return data;
}

export async function getDiseases(sessionId: string): Promise<{ enabled: boolean; diseases: DiseaseInfo[] }> {
  const { data } = await api.get(`/environment/${sessionId}/disease`);
  return data;
}

// === Social Hierarchy ===

export async function getHierarchy(sessionId: string): Promise<HierarchyOverview> {
  const { data } = await api.get(`/social/${sessionId}/hierarchy`);
  return data;
}

export async function getRoles(sessionId: string): Promise<RoleBreakdown> {
  const { data } = await api.get(`/social/${sessionId}/roles`);
  return data;
}

export async function getMentorship(sessionId: string): Promise<{ enabled: boolean; chains: MentorshipChain[] }> {
  const { data } = await api.get(`/social/${sessionId}/mentorship`);
  return data;
}

export async function getInfluenceMap(sessionId: string): Promise<{ enabled: boolean; agents: InfluenceEntry[] }> {
  const { data } = await api.get(`/social/${sessionId}/influence-map`);
  return data;
}

// === Genetics ===

export async function getAlleleFrequencies(sessionId: string): Promise<AlleleFrequencies> {
  const { data } = await api.get(`/genetics/${sessionId}/allele-frequencies`);
  return data;
}

export async function getEpigeneticPrevalence(sessionId: string): Promise<EpigeneticPrevalence> {
  const { data } = await api.get(`/genetics/${sessionId}/epigenetic-prevalence`);
  return data;
}

export async function getTraitGeneCorrelation(sessionId: string): Promise<TraitGeneCorrelation> {
  const { data } = await api.get(`/genetics/${sessionId}/trait-gene-correlation`);
  return data;
}

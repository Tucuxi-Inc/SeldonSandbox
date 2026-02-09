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
}): Promise<AgentSummary> {
  const { data } = await api.post('/experiments/inject-outsider', params);
  return data;
}

export async function getRippleReport(sessionId: string): Promise<Record<string, unknown>> {
  const { data } = await api.get(`/experiments/${sessionId}/ripple`);
  return data;
}

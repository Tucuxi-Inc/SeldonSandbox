import { useState, useEffect } from 'react';
import { useSimulationStore } from '../../../stores/simulation';
import { EmptyState } from '../../shared/EmptyState';
import { REGION_COLORS, SEVERITY_COLORS } from '../../../lib/constants';
import * as api from '../../../api/client';
import type { AgentSummary, AgentDetail, NotableEvent, LLMStatus } from '../../../types';
import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  BarChart,
  Bar,
  Cell,
} from 'recharts';

const REGION_LABELS: Record<string, string> = {
  under_processing: 'R1: Under-Processing',
  optimal: 'R2: Optimal',
  deep: 'R3: Deep',
  sacrificial: 'R4: Sacrificial',
  pathological: 'R5: Pathological',
};

export function BiographyView() {
  const { activeSessionId } = useSimulationStore();

  // Agent list
  const [allAgents, setAllAgents] = useState<AgentSummary[]>([]);
  const [search, setSearch] = useState('');
  const [loadingList, setLoadingList] = useState(false);

  // Selected agent state
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<AgentDetail | null>(null);
  const [timeline, setTimeline] = useState<NotableEvent[]>([]);
  const [loadingDetail, setLoadingDetail] = useState(false);

  // LLM prose
  const [prose, setProse] = useState<string | null>(null);
  const [loadingProse, setLoadingProse] = useState(false);

  // LLM provider state
  const [llmStatus, setLlmStatus] = useState<LLMStatus | null>(null);
  const [provider, setProvider] = useState<string>('anthropic');
  const [selectedModel, setSelectedModel] = useState<string | undefined>(undefined);

  // Fetch LLM status
  useEffect(() => {
    api.getLLMStatus().then((status) => {
      setLlmStatus(status);
      if (!status.providers.anthropic.available && status.providers.ollama.available) {
        setProvider('ollama');
        const models = status.providers.ollama.models ?? [];
        if (models.length > 0) setSelectedModel(models[0]);
      }
    }).catch(() => {});
  }, []);

  // Fetch all agents (alive + dead)
  useEffect(() => {
    if (!activeSessionId) {
      setAllAgents([]);
      return;
    }
    setLoadingList(true);
    api
      .listAgents(activeSessionId, { alive_only: false, page_size: 200 })
      .then((result) => setAllAgents(result.agents))
      .catch(() => setAllAgents([]))
      .finally(() => setLoadingList(false));
  }, [activeSessionId]);

  // Fetch detail + timeline when agent selected
  useEffect(() => {
    if (!activeSessionId || !selectedId) {
      setDetail(null);
      setTimeline([]);
      setProse(null);
      return;
    }
    setLoadingDetail(true);
    setProse(null);
    Promise.all([
      api.getAgentDetail(activeSessionId, selectedId),
      api.getAgentTimeline(activeSessionId, selectedId).catch(() => ({ events: [] })),
    ])
      .then(([d, t]) => {
        setDetail(d);
        setTimeline(t.events);
      })
      .catch(() => {
        setDetail(null);
        setTimeline([]);
      })
      .finally(() => setLoadingDetail(false));
  }, [activeSessionId, selectedId]);

  // Handle LLM prose generation
  const handleGenerateProse = async () => {
    if (!activeSessionId || !selectedId) return;
    setLoadingProse(true);
    try {
      const bio = await api.getAgentBiography(activeSessionId, selectedId, true, provider, selectedModel);
      if (bio.prose_error) {
        setProse(`LLM error: ${bio.prose_error}`);
      } else {
        setProse(bio.prose ?? 'No prose was generated.');
      }
    } catch {
      setProse('Failed to generate biography prose. Check LLM settings.');
    } finally {
      setLoadingProse(false);
    }
  };

  // Filter agents by search
  const filtered = allAgents.filter((a) => {
    if (!search) return true;
    const q = search.toLowerCase();
    return a.name.toLowerCase().includes(q) || a.id.includes(q);
  });

  if (!activeSessionId) {
    return <EmptyState message="Create a session to view agent biographies" />;
  }

  if (loadingList && allAgents.length === 0) {
    return <div className="p-6 text-gray-400">Loading agents...</div>;
  }

  if (allAgents.length === 0) {
    return <EmptyState message="Run a simulation to view agent biographies" />;
  }

  // Prepare radar data: top 8 traits by value
  const radarData = detail
    ? Object.entries(detail.traits)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 8)
        .map(([name, value]) => ({
          trait: name.replace(/_/g, ' '),
          value: Math.round(value * 1000) / 1000,
          birth: detail.traits_at_birth[name] != null
            ? Math.round(detail.traits_at_birth[name] * 1000) / 1000
            : 0,
        }))
    : [];

  // Contribution arc data
  const contributionData = detail
    ? detail.contribution_history.map((c, i) => ({
        gen: i,
        contribution: Math.round(c * 1000) / 1000,
      }))
    : [];

  // Death analysis bar data
  const deathInfo = detail
    ? (detail as AgentDetail & {
        death_info?: {
          mortality_breakdown: Record<string, number>;
          primary_cause: string;
          age_at_death: number;
          processing_region_at_death: string;
          suffering_at_death: number;
          burnout_at_death: number;
        };
      }).death_info
    : null;

  const mortalityBarData = deathInfo?.mortality_breakdown
    ? Object.entries(deathInfo.mortality_breakdown)
        .map(([factor, value]) => ({
          factor: factor.replace(/_/g, ' '),
          value: Math.round((value as number) * 10000) / 10000,
        }))
        .sort((a, b) => b.value - a.value)
    : [];

  // Lookup agent name by id
  const agentName = (id: string): string => {
    const a = allAgents.find((ag) => ag.id === id);
    return a ? a.name : id;
  };

  return (
    <div className="flex gap-4 h-[calc(100vh-8rem)]">
      {/* Agent Selector Panel */}
      <div className="w-80 flex flex-col rounded-lg border border-gray-800 bg-gray-900 shrink-0">
        <div className="p-3 border-b border-gray-800">
          <input
            type="text"
            placeholder="Search agents by name or ID..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full rounded-md border border-gray-700 bg-gray-800 px-3 py-1.5 text-sm text-gray-200 placeholder-gray-500 focus:border-blue-500 focus:outline-none"
          />
          <div className="mt-1 text-xs text-gray-500">
            {filtered.length} agents (alive + dead)
          </div>
        </div>
        <div className="flex-1 overflow-y-auto">
          {filtered.slice(0, 200).map((agent) => (
            <button
              key={agent.id}
              onClick={() => setSelectedId(agent.id)}
              className={`w-full text-left px-3 py-2 border-b border-gray-800/50 hover:bg-gray-800/50 transition-colors ${
                selectedId === agent.id ? 'bg-gray-800' : ''
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-gray-200 truncate">
                  {agent.name}
                </span>
                <div className="flex items-center gap-1.5 shrink-0">
                  {!agent.is_alive && (
                    <span className="rounded-full bg-red-900/50 px-1.5 py-0.5 text-[10px] text-red-300 border border-red-800">
                      Dead
                    </span>
                  )}
                  <span
                    className="rounded-full px-1.5 py-0.5 text-[10px] font-medium border"
                    style={{
                      backgroundColor: `${REGION_COLORS[agent.processing_region] || '#6B7280'}20`,
                      borderColor: `${REGION_COLORS[agent.processing_region] || '#6B7280'}60`,
                      color: REGION_COLORS[agent.processing_region] || '#6B7280',
                    }}
                  >
                    {REGION_LABELS[agent.processing_region] || agent.processing_region}
                  </span>
                </div>
              </div>
              <div className="text-xs text-gray-500 mt-0.5">
                Age {agent.age} | Gen {agent.generation}
                {agent.is_outsider && ' | Outsider'}
              </div>
            </button>
          ))}
          {filtered.length > 200 && (
            <div className="p-3 text-xs text-gray-600 text-center">
              Showing 200 of {filtered.length} agents
            </div>
          )}
        </div>
      </div>

      {/* Detail Panel */}
      <div className="flex-1 overflow-y-auto">
        {!selectedId || loadingDetail ? (
          loadingDetail ? (
            <div className="p-6 text-gray-400">Loading agent biography...</div>
          ) : (
            <EmptyState message="Select an agent to view their biography" />
          )
        ) : !detail ? (
          <EmptyState message="Agent data unavailable" />
        ) : (
          <div className="space-y-4 pb-6">
            {/* Biography Header */}
            <div className="rounded-lg border border-gray-800 bg-gray-900 p-5">
              <div className="flex items-center gap-3">
                <h2 className="text-2xl font-bold text-gray-100">{detail.name}</h2>
                {detail.is_alive ? (
                  <span className="rounded-full bg-emerald-900/50 px-2.5 py-0.5 text-xs font-medium text-emerald-300 border border-emerald-700">
                    Alive
                  </span>
                ) : (
                  <span className="rounded-full bg-red-900/50 px-2.5 py-0.5 text-xs font-medium text-red-300 border border-red-800">
                    Dead
                    {deathInfo ? ` (age ${deathInfo.age_at_death})` : ''}
                  </span>
                )}
                {detail.is_outsider && (
                  <span className="rounded-full bg-amber-900/50 px-2.5 py-0.5 text-xs font-medium text-amber-300 border border-amber-700">
                    Outsider
                  </span>
                )}
              </div>
              <div className="mt-3 grid grid-cols-2 sm:grid-cols-4 gap-3 text-sm">
                <div>
                  <span className="text-gray-500">Age:</span>{' '}
                  <span className="text-gray-200">{detail.age}</span>
                </div>
                <div>
                  <span className="text-gray-500">Generation:</span>{' '}
                  <span className="text-gray-200">{detail.generation}</span>
                </div>
                <div>
                  <span className="text-gray-500">Birth Order:</span>{' '}
                  <span className="text-gray-200">{detail.birth_order}</span>
                </div>
                <div>
                  <span className="text-gray-500">Status:</span>{' '}
                  <span className="text-gray-200">{detail.relationship_status}</span>
                </div>
                <div>
                  <span className="text-gray-500">Burnout:</span>{' '}
                  <span className="text-gray-200">{detail.burnout_level.toFixed(3)}</span>
                </div>
                <div>
                  <span className="text-gray-500">Suffering:</span>{' '}
                  <span className="text-gray-200">{detail.suffering.toFixed(3)}</span>
                </div>
                <div>
                  <span className="text-gray-500">Contribution:</span>{' '}
                  <span className="text-gray-200">{detail.latest_contribution.toFixed(3)}</span>
                </div>
                <div>
                  <span className="text-gray-500">ID:</span>{' '}
                  <span className="font-mono text-gray-400 text-xs">{detail.id}</span>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
              {/* Personality Profile (Radar) */}
              <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wide">
                    Personality Profile
                  </h3>
                  <span
                    className="rounded-full px-2 py-0.5 text-xs font-medium border"
                    style={{
                      backgroundColor: `${REGION_COLORS[detail.processing_region] || '#6B7280'}20`,
                      borderColor: `${REGION_COLORS[detail.processing_region] || '#6B7280'}60`,
                      color: REGION_COLORS[detail.processing_region] || '#6B7280',
                    }}
                  >
                    {REGION_LABELS[detail.processing_region] || detail.processing_region}
                  </span>
                </div>
                {radarData.length > 0 ? (
                  <ResponsiveContainer width="100%" height={300}>
                    <RadarChart data={radarData}>
                      <PolarGrid stroke="#374151" />
                      <PolarAngleAxis
                        dataKey="trait"
                        tick={{ fill: '#9ca3af', fontSize: 10 }}
                      />
                      <Radar
                        name="Current"
                        dataKey="value"
                        stroke="#3B82F6"
                        fill="#3B82F6"
                        fillOpacity={0.3}
                      />
                      <Radar
                        name="At Birth"
                        dataKey="birth"
                        stroke="#F59E0B"
                        fill="#F59E0B"
                        fillOpacity={0.1}
                      />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: '#111827',
                          border: '1px solid #374151',
                          borderRadius: '8px',
                        }}
                        formatter={(value: unknown) =>
                          (typeof value === 'number' ? value.toFixed(3) : value) as string
                        }
                      />
                    </RadarChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="h-[300px] flex items-center justify-center text-gray-500 text-sm">
                    No trait data available
                  </div>
                )}
                {detail.dominant_voice && (
                  <div className="mt-2 text-xs text-gray-400">
                    <span className="text-gray-500">Dominant Voice:</span>{' '}
                    {detail.dominant_voice}
                  </div>
                )}
              </div>

              {/* Contribution Arc */}
              <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
                <h3 className="mb-2 text-sm font-semibold text-gray-400 uppercase tracking-wide">
                  Contribution Arc
                </h3>
                {contributionData.length > 0 ? (
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={contributionData}>
                      <XAxis
                        dataKey="gen"
                        stroke="#6b7280"
                        tick={{ fontSize: 11 }}
                        label={{
                          value: 'Generation',
                          position: 'insideBottom',
                          offset: -5,
                          fill: '#6b7280',
                        }}
                      />
                      <YAxis stroke="#6b7280" tick={{ fontSize: 11 }} />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: '#111827',
                          border: '1px solid #374151',
                          borderRadius: '8px',
                        }}
                        formatter={(value: unknown) =>
                          (typeof value === 'number' ? value.toFixed(3) : value) as string
                        }
                      />
                      <Line
                        type="monotone"
                        dataKey="contribution"
                        stroke="#10B981"
                        strokeWidth={2}
                        dot={contributionData.length < 30}
                        name="Contribution"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="h-[300px] flex items-center justify-center text-gray-500 text-sm">
                    No contribution history
                  </div>
                )}
              </div>
            </div>

            {/* Region Journey */}
            {detail.region_history.length > 0 && (
              <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
                <h3 className="mb-2 text-sm font-semibold text-gray-400 uppercase tracking-wide">
                  Processing Region Journey
                </h3>
                <div className="flex h-8 rounded overflow-hidden">
                  {detail.region_history.map((region, i) => (
                    <div
                      key={i}
                      className="flex-1"
                      style={{
                        backgroundColor: REGION_COLORS[region] || '#6B7280',
                      }}
                      title={`Gen ${i}: ${REGION_LABELS[region] || region}`}
                    />
                  ))}
                </div>
                <div className="flex justify-between mt-1 text-xs text-gray-600">
                  <span>Gen 0</span>
                  <span>Gen {detail.region_history.length - 1}</span>
                </div>
              </div>
            )}

            {/* Life Timeline */}
            {timeline.length > 0 && (
              <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
                <h3 className="mb-3 text-sm font-semibold text-gray-400 uppercase tracking-wide">
                  Life Timeline
                </h3>
                <div className="relative ml-4">
                  {/* Vertical line */}
                  <div className="absolute left-2 top-0 bottom-0 w-px bg-gray-700" />
                  <div className="space-y-4">
                    {timeline
                      .sort((a, b) => a.generation - b.generation)
                      .map((event, i) => {
                        const color =
                          SEVERITY_COLORS[event.severity] || SEVERITY_COLORS.minor;
                        return (
                          <div key={i} className="relative pl-8">
                            {/* Severity dot */}
                            <div
                              className="absolute left-0 top-1 w-4 h-4 rounded-full border-2"
                              style={{
                                backgroundColor: `${color}30`,
                                borderColor: color,
                              }}
                            />
                            <div className="flex items-start gap-2">
                              <span className="text-xs text-gray-500 shrink-0 mt-0.5 w-12">
                                Gen {event.generation}
                              </span>
                              <div className="flex-1 min-w-0">
                                <div className="flex items-center gap-2">
                                  <span className="text-sm font-medium text-gray-200">
                                    {event.headline}
                                  </span>
                                  <span
                                    className="rounded-full px-1.5 py-0.5 text-[10px] font-medium shrink-0"
                                    style={{
                                      backgroundColor: `${color}20`,
                                      color: color,
                                    }}
                                  >
                                    {event.severity}
                                  </span>
                                </div>
                                {event.detail && (
                                  <p className="text-xs text-gray-400 mt-0.5">
                                    {event.detail}
                                  </p>
                                )}
                                <span className="text-[10px] text-gray-600">
                                  {event.event_type.replace(/_/g, ' ')}
                                </span>
                              </div>
                            </div>
                          </div>
                        );
                      })}
                  </div>
                </div>
              </div>
            )}

            {/* Relationship Panel */}
            <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
              <h3 className="mb-2 text-sm font-semibold text-gray-400 uppercase tracking-wide">
                Relationships
              </h3>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-sm">
                {/* Partner */}
                <div>
                  <div className="text-xs text-gray-500 mb-1">Partner</div>
                  {detail.partner_id ? (
                    <button
                      onClick={() => setSelectedId(detail.partner_id!)}
                      className="text-blue-400 hover:underline"
                    >
                      {agentName(detail.partner_id)}
                    </button>
                  ) : (
                    <span className="text-gray-600">None</span>
                  )}
                </div>
                {/* Parents */}
                <div>
                  <div className="text-xs text-gray-500 mb-1">Parents</div>
                  {detail.parent1_id || detail.parent2_id ? (
                    <div className="space-y-0.5">
                      {detail.parent1_id && (
                        <button
                          onClick={() => setSelectedId(detail.parent1_id!)}
                          className="block text-blue-400 hover:underline"
                        >
                          {agentName(detail.parent1_id)}
                        </button>
                      )}
                      {detail.parent2_id && (
                        <button
                          onClick={() => setSelectedId(detail.parent2_id!)}
                          className="block text-blue-400 hover:underline"
                        >
                          {agentName(detail.parent2_id)}
                        </button>
                      )}
                    </div>
                  ) : (
                    <span className="text-gray-600">Founder</span>
                  )}
                </div>
                {/* Children */}
                <div>
                  <div className="text-xs text-gray-500 mb-1">
                    Children ({detail.children_ids.length})
                  </div>
                  {detail.children_ids.length > 0 ? (
                    <div className="space-y-0.5 max-h-32 overflow-y-auto">
                      {detail.children_ids.map((cid) => (
                        <button
                          key={cid}
                          onClick={() => setSelectedId(cid)}
                          className="block text-blue-400 hover:underline text-xs"
                        >
                          {agentName(cid)}
                        </button>
                      ))}
                    </div>
                  ) : (
                    <span className="text-gray-600">None</span>
                  )}
                </div>
              </div>
            </div>

            {/* Death Analysis */}
            {!detail.is_alive && deathInfo && (
              <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
                <h3 className="mb-3 text-sm font-semibold text-gray-400 uppercase tracking-wide">
                  Death Analysis
                </h3>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-sm mb-4">
                  <div>
                    <span className="text-gray-500">Primary Cause:</span>{' '}
                    <span className="text-gray-200">
                      {deathInfo.primary_cause.replace(/_/g, ' ')}
                    </span>
                  </div>
                  <div>
                    <span className="text-gray-500">Age at Death:</span>{' '}
                    <span className="text-gray-200">{deathInfo.age_at_death}</span>
                  </div>
                  <div>
                    <span className="text-gray-500">Region:</span>{' '}
                    <span
                      style={{
                        color:
                          REGION_COLORS[deathInfo.processing_region_at_death] ||
                          '#6B7280',
                      }}
                    >
                      {REGION_LABELS[deathInfo.processing_region_at_death] ||
                        deathInfo.processing_region_at_death}
                    </span>
                  </div>
                  <div>
                    <span className="text-gray-500">Suffering:</span>{' '}
                    <span className="text-gray-200">
                      {deathInfo.suffering_at_death.toFixed(3)}
                    </span>
                  </div>
                </div>
                {mortalityBarData.length > 0 && (
                  <ResponsiveContainer width="100%" height={Math.max(150, mortalityBarData.length * 32)}>
                    <BarChart
                      data={mortalityBarData}
                      layout="vertical"
                      margin={{ left: 100, right: 20, top: 5, bottom: 5 }}
                    >
                      <XAxis
                        type="number"
                        stroke="#6b7280"
                        tick={{ fontSize: 11 }}
                      />
                      <YAxis
                        type="category"
                        dataKey="factor"
                        stroke="#6b7280"
                        tick={{ fontSize: 11 }}
                        width={95}
                      />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: '#111827',
                          border: '1px solid #374151',
                          borderRadius: '8px',
                        }}
                        formatter={(value: unknown) =>
                          (typeof value === 'number' ? value.toFixed(4) : value) as string
                        }
                      />
                      <Bar dataKey="value" name="Mortality Factor" radius={[0, 4, 4, 0]}>
                        {mortalityBarData.map((entry, index) => (
                          <Cell
                            key={index}
                            fill={
                              entry.factor === deathInfo.primary_cause.replace(/_/g, ' ')
                                ? '#EF4444'
                                : '#3B82F6'
                            }
                          />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                )}
              </div>
            )}

            {/* LLM Prose Generation */}
            <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wide">
                  Narrative Biography
                </h3>
                <div className="flex items-center gap-2">
                  {/* Provider selector */}
                  <select
                    value={provider}
                    onChange={(e) => {
                      setProvider(e.target.value);
                      setSelectedModel(undefined);
                    }}
                    className="rounded-md border border-gray-700 bg-gray-800 px-2 py-1 text-xs text-gray-300"
                  >
                    <option value="anthropic">Anthropic</option>
                    <option value="ollama">Ollama</option>
                  </select>
                  {/* Model selector */}
                  {llmStatus && (
                    <select
                      value={selectedModel ?? ''}
                      onChange={(e) => setSelectedModel(e.target.value || undefined)}
                      className="rounded-md border border-gray-700 bg-gray-800 px-2 py-1 text-xs text-gray-300"
                    >
                      <option value="">Default</option>
                      {(provider === 'anthropic'
                        ? llmStatus.providers.anthropic.models ?? []
                        : llmStatus.providers.ollama.models ?? []
                      ).map((m) => (
                        <option key={m} value={m}>{m}</option>
                      ))}
                    </select>
                  )}
                  <button
                    onClick={handleGenerateProse}
                    disabled={loadingProse}
                    className="rounded-md bg-blue-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  >
                    {loadingProse ? 'Generating...' : 'Generate with LLM'}
                  </button>
                </div>
              </div>
              {prose ? (
                <div className="prose prose-invert prose-sm max-w-none">
                  <p className="text-gray-300 whitespace-pre-wrap leading-relaxed text-sm">
                    {prose}
                  </p>
                </div>
              ) : (
                <p className="text-xs text-gray-500">
                  Click the button to generate a narrative biography using the LLM provider.
                  Requires LLM to be configured in Settings.
                </p>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

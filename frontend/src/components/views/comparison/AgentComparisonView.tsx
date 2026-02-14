import { useState, useEffect } from 'react';
import { useSimulationStore } from '../../../stores/simulation';
import { EmptyState } from '../../shared/EmptyState';
import { REGION_COLORS, COMPARISON_COLORS } from '../../../lib/constants';
import * as api from '../../../api/client';
import type { AgentSummary, AgentDetail } from '../../../types';
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
  Legend,
} from 'recharts';

interface AgentSlot {
  id: string | null;
  detail: AgentDetail | null;
  loading: boolean;
}

interface RelationshipLink {
  type: string;
  agentA: string;
  agentB: string;
  detail: string;
}

const SLOT_COUNT = 3;

export function AgentComparisonView() {
  const { activeSessionId } = useSimulationStore();
  const [allAgents, setAllAgents] = useState<AgentSummary[]>([]);
  const [slots, setSlots] = useState<AgentSlot[]>(
    Array.from({ length: SLOT_COUNT }, () => ({ id: null, detail: null, loading: false })),
  );
  const [timelineMode, setTimelineMode] = useState<'contribution' | 'suffering'>('contribution');
  const [loadingAgents, setLoadingAgents] = useState(false);

  // Fetch all agents for selection dropdowns
  useEffect(() => {
    if (!activeSessionId) return;
    let cancelled = false;
    setLoadingAgents(true);
    api
      .listAgents(activeSessionId, { alive_only: false, page_size: 200 })
      .then((result) => {
        if (!cancelled) setAllAgents(result.agents);
      })
      .catch(() => {})
      .finally(() => {
        if (!cancelled) setLoadingAgents(false);
      });
    return () => {
      cancelled = true;
    };
  }, [activeSessionId]);

  // Reset slots when session changes
  useEffect(() => {
    setSlots(Array.from({ length: SLOT_COUNT }, () => ({ id: null, detail: null, loading: false })));
  }, [activeSessionId]);

  const handleSlotChange = async (slotIndex: number, agentId: string | null) => {
    if (!activeSessionId) return;

    setSlots((prev) => {
      const next = [...prev];
      next[slotIndex] = { id: agentId, detail: null, loading: !!agentId };
      return next;
    });

    if (!agentId) return;

    try {
      const detail = await api.getAgentDetail(activeSessionId, agentId);
      setSlots((prev) => {
        const next = [...prev];
        if (next[slotIndex].id === agentId) {
          next[slotIndex] = { id: agentId, detail, loading: false };
        }
        return next;
      });
    } catch {
      setSlots((prev) => {
        const next = [...prev];
        if (next[slotIndex].id === agentId) {
          next[slotIndex] = { id: agentId, detail: null, loading: false };
        }
        return next;
      });
    }
  };

  const selectedAgents = slots.filter((s) => s.detail !== null) as {
    id: string;
    detail: AgentDetail;
    loading: false;
  }[];

  // Build radar data: merge all agents' traits into unified data points
  const radarData = (() => {
    if (selectedAgents.length === 0) return [];
    const traitNames = Object.keys(selectedAgents[0].detail.traits);
    return traitNames.map((trait) => {
      const point: Record<string, string | number> = { trait };
      selectedAgents.forEach((agent, i) => {
        point[`agent_${i}`] = agent.detail.traits[trait] ?? 0;
      });
      return point;
    });
  })();

  // Build timeline data: x-axis = generation index, values per agent
  const timelineData = (() => {
    if (selectedAgents.length === 0) return [];
    const maxLen = Math.max(
      ...selectedAgents.map((a) =>
        timelineMode === 'contribution'
          ? a.detail.contribution_history.length
          : a.detail.suffering_history.length,
      ),
    );
    const rows: Record<string, number>[] = [];
    for (let gen = 0; gen < maxLen; gen++) {
      const row: Record<string, number> = { gen };
      selectedAgents.forEach((agent, i) => {
        const history =
          timelineMode === 'contribution'
            ? agent.detail.contribution_history
            : agent.detail.suffering_history;
        if (gen < history.length) {
          row[`agent_${i}`] = history[gen];
        }
      });
      rows.push(row);
    }
    return rows;
  })();

  // Find relationships between selected agents
  const relationships: RelationshipLink[] = (() => {
    const links: RelationshipLink[] = [];
    for (let i = 0; i < selectedAgents.length; i++) {
      for (let j = i + 1; j < selectedAgents.length; j++) {
        const a = selectedAgents[i].detail;
        const b = selectedAgents[j].detail;

        // Check parent-child
        if (a.parent1_id === b.id || a.parent2_id === b.id) {
          links.push({
            type: 'parent-child',
            agentA: b.name,
            agentB: a.name,
            detail: `${b.name} is a parent of ${a.name}`,
          });
        }
        if (b.parent1_id === a.id || b.parent2_id === a.id) {
          links.push({
            type: 'parent-child',
            agentA: a.name,
            agentB: b.name,
            detail: `${a.name} is a parent of ${b.name}`,
          });
        }

        // Check partners
        if (a.partner_id === b.id) {
          links.push({
            type: 'partners',
            agentA: a.name,
            agentB: b.name,
            detail: `${a.name} and ${b.name} are partners`,
          });
        }

        // Check siblings (share a parent)
        const aParents = [a.parent1_id, a.parent2_id].filter(Boolean);
        const bParents = [b.parent1_id, b.parent2_id].filter(Boolean);
        const sharedParents = aParents.filter((p) => bParents.includes(p));
        if (sharedParents.length > 0) {
          links.push({
            type: 'siblings',
            agentA: a.name,
            agentB: b.name,
            detail: `${a.name} and ${b.name} are siblings (${sharedParents.length} shared parent${sharedParents.length > 1 ? 's' : ''})`,
          });
        }

        // Check if one is a child of the other
        if (a.children_ids.includes(b.id)) {
          // Already covered by parent-child above, skip duplicate
        } else if (b.children_ids.includes(a.id)) {
          // Already covered
        }
      }
    }
    return links;
  })();

  // Compute max region history length across selected agents
  const maxRegionLen = selectedAgents.length > 0
    ? Math.max(...selectedAgents.map((a) => a.detail.region_history.length))
    : 0;

  if (!activeSessionId) {
    return <EmptyState message="Select a session to compare agents" />;
  }

  if (loadingAgents) {
    return <EmptyState message="Loading agents..." />;
  }

  if (allAgents.length === 0) {
    return <EmptyState message="Run a simulation to compare agents" />;
  }

  return (
    <div className="space-y-4">
      {/* Agent Selectors */}
      <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
        <h3 className="mb-3 text-sm font-semibold text-gray-400 uppercase tracking-wide">
          Select Agents to Compare
        </h3>
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
          {slots.map((slot, i) => {
            // Agents already selected in other slots
            const usedIds = slots
              .filter((_, idx) => idx !== i)
              .map((s) => s.id)
              .filter(Boolean) as string[];

            return (
              <div key={i} className="flex items-center gap-2">
                <div
                  className="w-3 h-3 rounded-full shrink-0"
                  style={{ backgroundColor: COMPARISON_COLORS[i] }}
                />
                <select
                  value={slot.id || ''}
                  onChange={(e) => handleSlotChange(i, e.target.value || null)}
                  className="flex-1 rounded-md border border-gray-700 bg-gray-800 px-3 py-1.5 text-sm text-gray-200 focus:border-blue-500 focus:outline-none"
                >
                  <option value="">
                    {i < 2 ? `Agent ${i + 1} (required)` : `Agent ${i + 1} (optional)`}
                  </option>
                  {allAgents
                    .filter((a) => !usedIds.includes(a.id))
                    .map((a) => (
                      <option key={a.id} value={a.id}>
                        {a.name} ({a.processing_region}
                        {a.is_alive ? '' : ', dead'})
                      </option>
                    ))}
                </select>
                {slot.loading && (
                  <span className="text-xs text-gray-500 animate-pulse">Loading...</span>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {selectedAgents.length < 2 && (
        <EmptyState message="Select at least 2 agents to begin comparison" />
      )}

      {selectedAgents.length >= 2 && (
        <>
          {/* Summary Cards */}
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
            {selectedAgents.map((agent, i) => (
              <div
                key={agent.id}
                className="rounded-lg border bg-gray-900 p-4"
                style={{ borderColor: COMPARISON_COLORS[i] }}
              >
                <div className="flex items-center gap-2 mb-2">
                  <div
                    className="w-3 h-3 rounded-full shrink-0"
                    style={{ backgroundColor: COMPARISON_COLORS[i] }}
                  />
                  <h4 className="text-sm font-bold text-gray-100 truncate">
                    {agent.detail.name}
                  </h4>
                </div>
                <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
                  <div>
                    <span className="text-gray-500">Age:</span>{' '}
                    <span className="text-gray-200">{agent.detail.age}</span>
                  </div>
                  <div>
                    <span className="text-gray-500">Region:</span>{' '}
                    <span className="text-gray-200">{agent.detail.processing_region}</span>
                  </div>
                  <div>
                    <span className="text-gray-500">Birth Order:</span>{' '}
                    <span className="text-gray-200">{agent.detail.birth_order}</span>
                  </div>
                  <div>
                    <span className="text-gray-500">Alive:</span>{' '}
                    <span className="text-gray-200">{agent.detail.is_alive ? 'Yes' : 'No'}</span>
                  </div>
                  <div>
                    <span className="text-gray-500">Suffering:</span>{' '}
                    <span className="text-gray-200">{agent.detail.suffering.toFixed(3)}</span>
                  </div>
                  <div>
                    <span className="text-gray-500">Burnout:</span>{' '}
                    <span className="text-gray-200">{agent.detail.burnout_level.toFixed(3)}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Radar Chart */}
          <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
            <h3 className="mb-2 text-sm font-semibold text-gray-400 uppercase tracking-wide">
              Trait Comparison
            </h3>
            <ResponsiveContainer width="100%" height={400}>
              <RadarChart data={radarData}>
                <PolarGrid stroke="#374151" />
                <PolarAngleAxis dataKey="trait" tick={{ fill: '#9ca3af', fontSize: 10 }} />
                {selectedAgents.map((agent, i) => (
                  <Radar
                    key={agent.id}
                    name={agent.detail.name}
                    dataKey={`agent_${i}`}
                    stroke={COMPARISON_COLORS[i]}
                    fill={COMPARISON_COLORS[i]}
                    fillOpacity={0.15}
                    strokeWidth={2}
                  />
                ))}
                <Legend wrapperStyle={{ fontSize: 12, color: '#9ca3af' }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#111827',
                    border: '1px solid #374151',
                    borderRadius: '8px',
                  }}
                  formatter={(_value: unknown) => {
                    const v = _value as number | undefined;
                    return v != null ? v.toFixed(3) : '';
                  }}
                />
              </RadarChart>
            </ResponsiveContainer>
          </div>

          {/* Timeline Comparison */}
          <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wide">
                Timeline Comparison
              </h3>
              <div className="flex gap-1">
                <button
                  onClick={() => setTimelineMode('contribution')}
                  className={`px-3 py-1 text-xs rounded-md transition-colors ${
                    timelineMode === 'contribution'
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                  }`}
                >
                  Contribution
                </button>
                <button
                  onClick={() => setTimelineMode('suffering')}
                  className={`px-3 py-1 text-xs rounded-md transition-colors ${
                    timelineMode === 'suffering'
                      ? 'bg-red-600 text-white'
                      : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                  }`}
                >
                  Suffering
                </button>
              </div>
            </div>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={timelineData}>
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
                  formatter={(_value: unknown) => {
                    const v = _value as number | undefined;
                    return v != null ? v.toFixed(3) : '';
                  }}
                />
                {selectedAgents.map((agent, i) => (
                  <Line
                    key={agent.id}
                    type="monotone"
                    dataKey={`agent_${i}`}
                    stroke={COMPARISON_COLORS[i]}
                    strokeWidth={2}
                    dot={false}
                    name={agent.detail.name}
                    connectNulls={false}
                  />
                ))}
                <Legend wrapperStyle={{ fontSize: 12, color: '#9ca3af' }} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Region Timeline */}
          <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
            <h3 className="mb-3 text-sm font-semibold text-gray-400 uppercase tracking-wide">
              Region Progression
            </h3>
            <div className="space-y-3">
              {selectedAgents.map((agent, i) => (
                <div key={agent.id}>
                  <div className="flex items-center gap-2 mb-1">
                    <div
                      className="w-2.5 h-2.5 rounded-full shrink-0"
                      style={{ backgroundColor: COMPARISON_COLORS[i] }}
                    />
                    <span className="text-xs text-gray-300 font-medium">{agent.detail.name}</span>
                  </div>
                  <div className="flex h-6 rounded overflow-hidden">
                    {agent.detail.region_history.map((region, genIdx) => (
                      <div
                        key={genIdx}
                        className="transition-colors"
                        style={{
                          backgroundColor: REGION_COLORS[region] || '#6B7280',
                          width: `${100 / maxRegionLen}%`,
                        }}
                        title={`Gen ${genIdx}: ${region}`}
                      />
                    ))}
                    {/* Fill remaining space if this agent has fewer generations */}
                    {agent.detail.region_history.length < maxRegionLen && (
                      <div
                        style={{
                          width: `${((maxRegionLen - agent.detail.region_history.length) / maxRegionLen) * 100}%`,
                          backgroundColor: '#1f2937',
                        }}
                      />
                    )}
                  </div>
                </div>
              ))}
            </div>
            <div className="flex justify-between mt-2 text-xs text-gray-600">
              <span>Gen 0</span>
              <span>Gen {maxRegionLen > 0 ? maxRegionLen - 1 : 0}</span>
            </div>
            {/* Region legend */}
            <div className="flex flex-wrap gap-3 mt-3">
              {Object.entries(REGION_COLORS).map(([key, color]) => (
                <div key={key} className="flex items-center gap-1.5">
                  <div
                    className="w-3 h-3 rounded-sm"
                    style={{ backgroundColor: color }}
                  />
                  <span className="text-xs text-gray-500">{key.replace(/_/g, ' ')}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Relationships */}
          <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
            <h3 className="mb-2 text-sm font-semibold text-gray-400 uppercase tracking-wide">
              Relationships Between Compared Agents
            </h3>
            {relationships.length === 0 ? (
              <p className="text-sm text-gray-500">
                No direct relationships found between the selected agents.
              </p>
            ) : (
              <div className="space-y-2">
                {relationships.map((rel, i) => (
                  <div
                    key={i}
                    className="flex items-center gap-3 rounded-md bg-gray-950 border border-gray-800 px-3 py-2"
                  >
                    <span
                      className={`inline-block rounded-full px-2 py-0.5 text-xs font-medium ${
                        rel.type === 'partners'
                          ? 'bg-red-900/50 text-red-300 border border-red-700'
                          : rel.type === 'parent-child'
                            ? 'bg-green-900/50 text-green-300 border border-green-700'
                            : 'bg-blue-900/50 text-blue-300 border border-blue-700'
                      }`}
                    >
                      {rel.type}
                    </span>
                    <span className="text-sm text-gray-200">{rel.detail}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}

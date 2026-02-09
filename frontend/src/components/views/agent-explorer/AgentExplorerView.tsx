import { useState } from 'react';
import { useSimulationStore } from '../../../stores/simulation';
import { EmptyState } from '../../shared/EmptyState';
import { RegionBadge } from '../../shared/RegionBadge';
import { REGION_COLORS } from '../../../lib/constants';
import * as api from '../../../api/client';
import type { AgentDetail } from '../../../types';
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  Legend,
} from 'recharts';

export function AgentExplorerView() {
  const { activeSessionId, agents } = useSimulationStore();
  const [search, setSearch] = useState('');
  const [regionFilter, setRegionFilter] = useState('');
  const [selectedDetail, setSelectedDetail] = useState<AgentDetail | null>(null);

  const filtered = agents.filter((a) => {
    if (search && !a.name.toLowerCase().includes(search.toLowerCase()) && !a.id.includes(search)) return false;
    if (regionFilter && a.processing_region !== regionFilter) return false;
    return true;
  });

  const handleSelect = async (agentId: string) => {
    if (!activeSessionId) return;
    const detail = await api.getAgentDetail(activeSessionId, agentId);
    setSelectedDetail(detail);
  };

  if (agents.length === 0) {
    return <EmptyState message="Run a simulation to explore agents" />;
  }

  // Trait radar data
  const radarData = selectedDetail
    ? Object.entries(selectedDetail.traits).map(([name, value]) => ({
        trait: name,
        current: value,
        birth: selectedDetail.traits_at_birth[name] || 0,
      }))
    : [];

  // Contribution/suffering history
  const historyData = selectedDetail
    ? selectedDetail.contribution_history.map((c, i) => ({
        gen: i,
        contribution: c,
        suffering: selectedDetail.suffering_history[i] || 0,
      }))
    : [];

  return (
    <div className="flex gap-4 h-[calc(100vh-8rem)]">
      {/* Agent List */}
      <div className="w-80 flex flex-col rounded-lg border border-gray-800 bg-gray-900">
        <div className="p-3 border-b border-gray-800 space-y-2">
          <input
            type="text"
            placeholder="Search agents..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full rounded-md border border-gray-700 bg-gray-800 px-3 py-1.5 text-sm text-gray-200 placeholder-gray-500 focus:border-blue-500 focus:outline-none"
          />
          <select
            value={regionFilter}
            onChange={(e) => setRegionFilter(e.target.value)}
            className="w-full rounded-md border border-gray-700 bg-gray-800 px-3 py-1.5 text-sm text-gray-200"
          >
            <option value="">All regions</option>
            <option value="under_processing">R1: Under-Processing</option>
            <option value="optimal">R2: Optimal</option>
            <option value="deep">R3: Deep</option>
            <option value="sacrificial">R4: Sacrificial</option>
            <option value="pathological">R5: Pathological</option>
          </select>
        </div>
        <div className="flex-1 overflow-y-auto">
          {filtered.slice(0, 100).map((agent) => (
            <button
              key={agent.id}
              onClick={() => handleSelect(agent.id)}
              className={`w-full text-left px-3 py-2 border-b border-gray-800/50 hover:bg-gray-800/50 transition-colors ${
                selectedDetail?.id === agent.id ? 'bg-gray-800' : ''
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-gray-200">{agent.name}</span>
                <RegionBadge region={agent.processing_region} />
              </div>
              <div className="text-xs text-gray-500 mt-0.5">
                Age {agent.age} | Birth order {agent.birth_order}
                {agent.is_outsider && ' | Outsider'}
              </div>
            </button>
          ))}
          {filtered.length > 100 && (
            <div className="p-3 text-xs text-gray-600 text-center">
              Showing 100 of {filtered.length} agents
            </div>
          )}
        </div>
      </div>

      {/* Detail Panel */}
      <div className="flex-1 overflow-y-auto">
        {!selectedDetail ? (
          <EmptyState message="Select an agent to view details" />
        ) : (
          <div className="space-y-4">
            {/* Header */}
            <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
              <div className="flex items-center gap-3">
                <h2 className="text-xl font-bold text-gray-100">{selectedDetail.name}</h2>
                <RegionBadge region={selectedDetail.processing_region} />
                {selectedDetail.is_outsider && (
                  <span className="rounded-full bg-amber-900/50 px-2 py-0.5 text-xs text-amber-300 border border-amber-700">
                    Outsider
                  </span>
                )}
              </div>
              <div className="mt-2 grid grid-cols-4 gap-4 text-sm">
                <div><span className="text-gray-500">ID:</span> <span className="font-mono text-gray-300">{selectedDetail.id}</span></div>
                <div><span className="text-gray-500">Age:</span> <span className="text-gray-200">{selectedDetail.age}</span></div>
                <div><span className="text-gray-500">Gen:</span> <span className="text-gray-200">{selectedDetail.generation}</span></div>
                <div><span className="text-gray-500">Birth Order:</span> <span className="text-gray-200">{selectedDetail.birth_order}</span></div>
                <div><span className="text-gray-500">Partner:</span> <span className="text-gray-200">{selectedDetail.partner_id || 'None'}</span></div>
                <div><span className="text-gray-500">Status:</span> <span className="text-gray-200">{selectedDetail.relationship_status}</span></div>
                <div><span className="text-gray-500">Children:</span> <span className="text-gray-200">{selectedDetail.children_ids.length}</span></div>
                <div><span className="text-gray-500">Burnout:</span> <span className="text-gray-200">{selectedDetail.burnout_level.toFixed(3)}</span></div>
              </div>
            </div>

            <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
              {/* Trait Radar */}
              <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
                <h3 className="mb-2 text-sm font-semibold text-gray-400 uppercase tracking-wide">Trait Profile</h3>
                <ResponsiveContainer width="100%" height={350}>
                  <RadarChart data={radarData}>
                    <PolarGrid stroke="#374151" />
                    <PolarAngleAxis dataKey="trait" tick={{ fill: '#9ca3af', fontSize: 10 }} />
                    <PolarRadiusAxis domain={[0, 1]} tick={{ fill: '#6b7280', fontSize: 10 }} />
                    <Radar name="Current" dataKey="current" stroke="#3B82F6" fill="#3B82F6" fillOpacity={0.3} />
                    <Radar name="At Birth" dataKey="birth" stroke="#F59E0B" fill="#F59E0B" fillOpacity={0.1} />
                    <Legend wrapperStyle={{ fontSize: 12, color: '#9ca3af' }} />
                  </RadarChart>
                </ResponsiveContainer>
              </div>

              {/* Contribution/Suffering History */}
              <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
                <h3 className="mb-2 text-sm font-semibold text-gray-400 uppercase tracking-wide">Life History</h3>
                <ResponsiveContainer width="100%" height={350}>
                  <LineChart data={historyData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                    <XAxis dataKey="gen" stroke="#6b7280" tick={{ fontSize: 11 }} label={{ value: 'Generation', position: 'insideBottom', offset: -5, fill: '#6b7280' }} />
                    <YAxis stroke="#6b7280" tick={{ fontSize: 11 }} />
                    <Tooltip contentStyle={{ backgroundColor: '#111827', border: '1px solid #374151', borderRadius: '8px' }} />
                    <Line type="monotone" dataKey="contribution" stroke="#10B981" strokeWidth={2} dot={false} name="Contribution" />
                    <Line type="monotone" dataKey="suffering" stroke="#EF4444" strokeWidth={2} dot={false} name="Suffering" />
                    <Legend wrapperStyle={{ fontSize: 12, color: '#9ca3af' }} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Region Timeline */}
            <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
              <h3 className="mb-2 text-sm font-semibold text-gray-400 uppercase tracking-wide">Region Timeline</h3>
              <div className="flex h-8 rounded overflow-hidden">
                {selectedDetail.region_history.map((region, i) => (
                  <div
                    key={i}
                    className="flex-1"
                    style={{ backgroundColor: REGION_COLORS[region] || '#6B7280' }}
                    title={`Gen ${i}: ${region}`}
                  />
                ))}
              </div>
              <div className="flex justify-between mt-1 text-xs text-gray-600">
                <span>Gen 0</span>
                <span>Gen {selectedDetail.region_history.length - 1}</span>
              </div>
            </div>

            {/* Family */}
            {(selectedDetail.parent1_id || selectedDetail.parent2_id || selectedDetail.children_ids.length > 0) && (
              <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
                <h3 className="mb-2 text-sm font-semibold text-gray-400 uppercase tracking-wide">Family</h3>
                <div className="text-sm space-y-1">
                  {selectedDetail.parent1_id && (
                    <div><span className="text-gray-500">Parent 1:</span> <button onClick={() => handleSelect(selectedDetail.parent1_id!)} className="text-blue-400 hover:underline">{selectedDetail.parent1_id}</button></div>
                  )}
                  {selectedDetail.parent2_id && (
                    <div><span className="text-gray-500">Parent 2:</span> <button onClick={() => handleSelect(selectedDetail.parent2_id!)} className="text-blue-400 hover:underline">{selectedDetail.parent2_id}</button></div>
                  )}
                  {selectedDetail.children_ids.length > 0 && (
                    <div>
                      <span className="text-gray-500">Children: </span>
                      {selectedDetail.children_ids.map((cid, i) => (
                        <span key={`${cid}-${i}`}>
                          {i > 0 && ', '}
                          <button onClick={() => handleSelect(cid)} className="text-blue-400 hover:underline">{cid}</button>
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

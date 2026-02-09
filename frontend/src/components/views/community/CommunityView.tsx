import { useState, useEffect } from 'react';
import { useSimulationStore } from '../../../stores/simulation';
import * as api from '../../../api/client';
import type { CommunityOverview } from '../../../types';
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  ResponsiveContainer, Tooltip, BarChart, Bar, XAxis, YAxis, CartesianGrid,
} from 'recharts';

const COMMUNITY_COLORS = ['#3b82f6', '#22c55e', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#84cc16'];

export function CommunityView() {
  const { activeSessionId } = useSimulationStore();
  const [data, setData] = useState<CommunityOverview | null>(null);
  const [diplomacy, setDiplomacy] = useState<Record<string, unknown> | null>(null);
  const [selectedCommunity, setSelectedCommunity] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!activeSessionId) return;
    setLoading(true);
    Promise.all([
      api.getCommunities(activeSessionId),
      api.getDiplomacy(activeSessionId).catch(() => null),
    ]).then(([c, d]) => {
      setData(c);
      setDiplomacy(d);
      if (c.communities.length > 0 && !selectedCommunity) {
        setSelectedCommunity(c.communities[0].id);
      }
    }).catch(() => {}).finally(() => setLoading(false));
  }, [activeSessionId]);

  if (!activeSessionId) {
    return <div className="p-6 text-gray-400">Create a session to view communities.</div>;
  }

  if (loading) return <div className="p-6 text-gray-400">Loading...</div>;
  if (!data?.enabled) {
    return <div className="p-6 text-gray-400">Community extensions not enabled. Enable geography/migration to see communities.</div>;
  }

  const selected = data.communities.find((c) => c.id === selectedCommunity);
  const radarTraits = selected?.trait_means
    ? Object.entries(selected.trait_means).map(([name, value]) => ({ trait: name, value: Math.round(value * 100) / 100 }))
    : [];

  const cohesionData = data.communities.map((c) => ({
    name: c.name,
    cohesion: Math.round(c.cohesion * 100) / 100,
    population: c.population,
  }));

  const relations = (data.diplomatic_relations ?? (diplomacy?.relations as any[])) ?? [];

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-gray-100">Communities & Diplomacy</h1>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        {/* Community list */}
        <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
          <h2 className="mb-3 text-lg font-semibold text-gray-200">Communities</h2>
          <div className="space-y-2 max-h-[60vh] overflow-y-auto">
            {data.communities.map((c, i) => (
              <button
                key={c.id}
                onClick={() => setSelectedCommunity(c.id)}
                className={`w-full rounded border p-3 text-left transition-colors ${
                  selectedCommunity === c.id
                    ? 'border-blue-600 bg-gray-800'
                    : 'border-gray-800 bg-gray-950 hover:bg-gray-800'
                }`}
              >
                <div className="flex items-center gap-2">
                  <div className="h-3 w-3 rounded-full" style={{ backgroundColor: COMMUNITY_COLORS[i % COMMUNITY_COLORS.length] }} />
                  <span className="text-sm font-medium text-gray-200">{c.name}</span>
                </div>
                <div className="mt-1 text-xs text-gray-500">
                  Pop: {c.population} | Cohesion: {c.cohesion.toFixed(2)} | {c.dominant_region}
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Right panels */}
        <div className="lg:col-span-2 space-y-4">
          {/* Personality Radar */}
          {selected && radarTraits.length > 0 && (
            <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
              <h2 className="mb-3 text-lg font-semibold text-gray-200">{selected.name} — Personality Profile</h2>
              <ResponsiveContainer width="100%" height={300}>
                <RadarChart data={radarTraits}>
                  <PolarGrid stroke="#374151" />
                  <PolarAngleAxis dataKey="trait" tick={{ fontSize: 10, fill: '#9ca3af' }} />
                  <PolarRadiusAxis tick={{ fontSize: 10, fill: '#6b7280' }} domain={[0, 1]} />
                  <Radar dataKey="value" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.3} />
                  <Tooltip contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }} />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Cohesion Chart */}
          <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
            <h2 className="mb-3 text-lg font-semibold text-gray-200">Cohesion Scores</h2>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={cohesionData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="name" stroke="#9ca3af" tick={{ fontSize: 11 }} />
                <YAxis stroke="#9ca3af" domain={[0, 1]} tick={{ fontSize: 11 }} />
                <Tooltip contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }} />
                <Bar dataKey="cohesion" fill="#22c55e" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Diplomatic Relations */}
          {relations.length > 0 && (
            <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
              <h2 className="mb-3 text-lg font-semibold text-gray-200">Diplomatic Relations</h2>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-gray-800 text-left text-gray-500">
                      <th className="pb-2">Community A</th>
                      <th className="pb-2">Community B</th>
                      <th className="pb-2">Standing</th>
                      <th className="pb-2">Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {relations.map((r: any, i: number) => (
                      <tr key={i} className="border-b border-gray-800/50">
                        <td className="py-2 text-gray-300">{r.community_a}</td>
                        <td className="py-2 text-gray-300">{r.community_b}</td>
                        <td className="py-2 font-mono text-gray-300">{(r.standing ?? 0).toFixed(2)}</td>
                        <td className="py-2">
                          <span className={`rounded px-1.5 py-0.5 text-xs ${
                            r.status === 'alliance' ? 'bg-green-900 text-green-300' :
                            r.status === 'rivalry' ? 'bg-red-900 text-red-300' :
                            'bg-gray-800 text-gray-400'
                          }`}>
                            {r.status ?? 'neutral'}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

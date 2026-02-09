import { useState, useEffect } from 'react';
import {
  AreaChart, Area, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from 'recharts';
import { useSimulationStore } from '../../../stores/simulation';
import { EmptyState } from '../../shared/EmptyState';
import * as api from '../../../api/client';
import type { LoreOverview, MemePrevalence } from '../../../types';
import { MEMORY_TYPE_COLORS } from '../../../lib/constants';

export function LoreEvolutionView() {
  const { activeSessionId, metrics } = useSimulationStore();
  const [overview, setOverview] = useState<LoreOverview | null>(null);
  const [memes, setMemes] = useState<MemePrevalence | null>(null);

  useEffect(() => {
    if (!activeSessionId) return;
    api.getLoreOverview(activeSessionId).then(setOverview).catch(() => {});
    api.getMemePrevalence(activeSessionId).then(setMemes).catch(() => {});
  }, [activeSessionId, metrics.length]);

  if (!activeSessionId || metrics.length === 0) {
    return <EmptyState message="Run a simulation with lore enabled to see memory evolution" />;
  }

  if (!overview) {
    return <EmptyState message="Loading lore data..." />;
  }

  const ts = overview.time_series;
  const areaData = ts.generations.map((gen: number, i: number) => ({
    generation: gen,
    total: ts.total_memories[i] || 0,
    societal: ts.societal_memories[i] || 0,
    myths: ts.myths_count[i] || 0,
    fidelity: ts.mean_fidelity?.[i] ?? 1.0,
  }));

  // Build meme chart data
  const memeChartData = memes?.enabled && memes.generations
    ? memes.generations.map((gen: number, i: number) => {
        const point: Record<string, unknown> = { generation: gen };
        for (const [name, values] of Object.entries(memes.prevalence_over_time)) {
          point[name] = (values as number[])[i] ?? 0;
        }
        return point;
      })
    : null;

  const memeNames = memes?.enabled ? Object.keys(memes.prevalence_over_time) : [];
  const memeColors = ['#3B82F6', '#F59E0B', '#10B981', '#EF4444', '#8B5CF6', '#EC4899'];

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-gray-100">Lore Evolution</h1>

      {/* Memory counts area chart */}
      <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
        <h2 className="mb-4 text-lg font-semibold text-gray-200">Memory Counts Over Time</h2>
        <ResponsiveContainer width="100%" height={300}>
          <AreaChart data={areaData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="generation" stroke="#6b7280" fontSize={11} />
            <YAxis stroke="#6b7280" fontSize={11} />
            <Tooltip
              contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: '8px' }}
              labelStyle={{ color: '#9ca3af' }}
            />
            <Legend />
            <Area type="monotone" dataKey="total" stackId="1" stroke={MEMORY_TYPE_COLORS.personal} fill={MEMORY_TYPE_COLORS.personal} fillOpacity={0.6} name="Total Memories" />
            <Area type="monotone" dataKey="societal" stackId="2" stroke={MEMORY_TYPE_COLORS.societal} fill={MEMORY_TYPE_COLORS.societal} fillOpacity={0.6} name="Societal" />
            <Area type="monotone" dataKey="myths" stackId="3" stroke={MEMORY_TYPE_COLORS.myth} fill={MEMORY_TYPE_COLORS.myth} fillOpacity={0.6} name="Myths" />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Fidelity line */}
      <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
        <h2 className="mb-4 text-lg font-semibold text-gray-200">Mean Memory Fidelity</h2>
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={areaData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="generation" stroke="#6b7280" fontSize={11} />
            <YAxis domain={[0, 1]} stroke="#6b7280" fontSize={11} />
            <Tooltip
              contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: '8px' }}
              labelStyle={{ color: '#9ca3af' }}
              itemStyle={{ color: '#e5e7eb' }}
            />
            <Line type="monotone" dataKey="fidelity" stroke="#10B981" strokeWidth={2} dot={false} name="Fidelity" />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Meme prevalence */}
      {memes?.enabled && memeChartData && memeNames.length > 0 && (
        <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
          <h2 className="mb-4 text-lg font-semibold text-gray-200">Meme Prevalence</h2>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={memeChartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="generation" stroke="#6b7280" fontSize={11} />
              <YAxis domain={[0, 1]} stroke="#6b7280" fontSize={11} />
              <Tooltip
                contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: '8px' }}
                labelStyle={{ color: '#9ca3af' }}
              />
              <Legend />
              {memeNames.map((name, i) => (
                <Line
                  key={name}
                  type="monotone"
                  dataKey={name}
                  stroke={memeColors[i % memeColors.length]}
                  strokeWidth={2}
                  dot={false}
                  name={name.replace(/_/g, ' ')}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Memory type distribution */}
      <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
        <h2 className="mb-4 text-lg font-semibold text-gray-200">Current Memory Type Distribution</h2>
        <div className="grid grid-cols-4 gap-4">
          {Object.entries(overview.memory_type_distribution).map(([type, count]) => (
            <div key={type} className="text-center">
              <div className="text-2xl font-bold text-gray-100">{count}</div>
              <div className="flex items-center justify-center gap-1 text-sm text-gray-400">
                <div
                  className="h-2.5 w-2.5 rounded-full"
                  style={{ backgroundColor: MEMORY_TYPE_COLORS[type] || '#6B7280' }}
                />
                {type}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Current societal lore table */}
      {overview.current_societal_lore.length > 0 && (
        <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
          <h2 className="mb-4 text-lg font-semibold text-gray-200">
            Current Societal Lore ({overview.current_societal_lore.length})
          </h2>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-800 text-left text-gray-400">
                  <th className="px-3 py-2">Content</th>
                  <th className="px-3 py-2">Type</th>
                  <th className="px-3 py-2">Fidelity</th>
                  <th className="px-3 py-2">Valence</th>
                  <th className="px-3 py-2">Transmissions</th>
                </tr>
              </thead>
              <tbody>
                {overview.current_societal_lore.map((mem, i) => (
                  <tr key={i} className="border-b border-gray-800/50">
                    <td className="max-w-xs truncate px-3 py-2 text-gray-200">{mem.content}</td>
                    <td className="px-3 py-2 text-gray-400">{mem.memory_type}</td>
                    <td className="px-3 py-2">
                      <div className="flex items-center gap-2">
                        <div className="h-1.5 w-16 rounded-full bg-gray-700">
                          <div
                            className="h-1.5 rounded-full bg-green-500"
                            style={{ width: `${(mem.fidelity ?? 0) * 100}%` }}
                          />
                        </div>
                        <span className="text-xs text-gray-400">{((mem.fidelity ?? 0) * 100).toFixed(0)}%</span>
                      </div>
                    </td>
                    <td className="px-3 py-2">
                      <span className={`text-sm ${(mem.emotional_valence ?? 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                        {(mem.emotional_valence ?? 0) >= 0 ? '+' : ''}{(mem.emotional_valence ?? 0).toFixed(2)}
                      </span>
                    </td>
                    <td className="px-3 py-2 text-gray-400">{mem.transmission_count ?? 0}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

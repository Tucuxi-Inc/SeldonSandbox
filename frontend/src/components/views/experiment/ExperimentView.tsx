import { useState } from 'react';
import { useSimulationStore } from '../../../stores/simulation';
import { EmptyState } from '../../shared/EmptyState';
import * as api from '../../../api/client';
import type { ComparisonResponse, GenerationMetrics } from '../../../types';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from 'recharts';

const COMPARE_COLORS = ['#3B82F6', '#F59E0B', '#10B981', '#EF4444', '#8B5CF6'];

export function ExperimentView() {
  const { sessions } = useSimulationStore();
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [comparison, setComparison] = useState<ComparisonResponse | null>(null);
  const [metricsA, setMetricsA] = useState<GenerationMetrics[] | null>(null);
  const [metricsB, setMetricsB] = useState<GenerationMetrics[] | null>(null);
  const [loading, setLoading] = useState(false);

  const toggleSession = (id: string) => {
    setSelectedIds((prev) =>
      prev.includes(id) ? prev.filter((s) => s !== id) : [...prev, id],
    );
  };

  const handleCompare = async () => {
    if (selectedIds.length < 2) return;
    setLoading(true);
    const [comp, mA, mB] = await Promise.all([
      api.compareSessions(selectedIds.slice(0, 2)),
      api.getGenerations(selectedIds[0]),
      api.getGenerations(selectedIds[1]),
    ]);
    setComparison(comp);
    setMetricsA(mA);
    setMetricsB(mB);
    setLoading(false);
  };

  const completedSessions = sessions.filter((s) => s.status === 'completed');

  if (completedSessions.length < 2) {
    return <EmptyState message="Complete at least 2 simulations to compare experiments" />;
  }

  // Build overlaid data
  const maxGen = Math.max(
    metricsA?.length || 0,
    metricsB?.length || 0,
  );
  const overlaidData = Array.from({ length: maxGen }, (_, i) => ({
    generation: i,
    pop_a: metricsA?.[i]?.population_size ?? null,
    pop_b: metricsB?.[i]?.population_size ?? null,
    contrib_a: metricsA?.[i]?.mean_contribution ?? null,
    contrib_b: metricsB?.[i]?.mean_contribution ?? null,
    suffer_a: metricsA?.[i]?.mean_suffering ?? null,
    suffer_b: metricsB?.[i]?.mean_suffering ?? null,
  }));

  const sessionNames = selectedIds.map((id) => {
    const s = sessions.find((s) => s.id === id);
    return s ? `${s.name} (${id})` : id;
  });

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-gray-100">Experiment Comparison</h1>

      {/* Session Selector */}
      <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
        <h2 className="mb-3 text-sm font-semibold text-gray-400 uppercase tracking-wide">Select Sessions to Compare</h2>
        <div className="flex flex-wrap gap-2">
          {completedSessions.map((s) => (
            <button
              key={s.id}
              onClick={() => toggleSession(s.id)}
              className={`rounded-md px-3 py-1.5 text-sm border transition-colors ${
                selectedIds.includes(s.id)
                  ? 'border-blue-500 bg-blue-900/30 text-blue-300'
                  : 'border-gray-700 bg-gray-800 text-gray-300 hover:border-gray-600'
              }`}
            >
              {s.name} ({s.id})
            </button>
          ))}
        </div>
        <button
          onClick={handleCompare}
          disabled={selectedIds.length < 2 || loading}
          className="mt-3 rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-40"
        >
          {loading ? 'Comparing...' : 'Compare'}
        </button>
      </div>

      {comparison && metricsA && metricsB && (
        <>
          {/* Summary Stats */}
          <div className="grid grid-cols-2 gap-4">
            {selectedIds.slice(0, 2).map((id, idx) => {
              const stats = comparison.sessions[id];
              return (
                <div key={id} className="rounded-lg border bg-gray-900 p-4" style={{ borderColor: COMPARE_COLORS[idx] + '44' }}>
                  <h3 className="text-sm font-semibold" style={{ color: COMPARE_COLORS[idx] }}>{sessionNames[idx]}</h3>
                  <div className="mt-2 grid grid-cols-2 gap-2 text-sm">
                    <div><span className="text-gray-500">Final Pop:</span> <span className="text-gray-200">{stats.final_population_size}</span></div>
                    <div><span className="text-gray-500">Breakthroughs:</span> <span className="text-gray-200">{stats.total_breakthroughs}</span></div>
                    <div><span className="text-gray-500">Avg Contrib:</span> <span className="text-gray-200">{stats.mean_contribution.toFixed(4)}</span></div>
                    <div><span className="text-gray-500">Avg Suffering:</span> <span className="text-gray-200">{stats.mean_suffering.toFixed(4)}</span></div>
                    <div><span className="text-gray-500">Peak Pop:</span> <span className="text-gray-200">{stats.peak_population}</span></div>
                    <div><span className="text-gray-500">Generations:</span> <span className="text-gray-200">{stats.total_generations}</span></div>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Config Diffs */}
          {Object.keys(comparison.config_diffs).length > 0 && (
            <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
              <h2 className="mb-3 text-sm font-semibold text-gray-400 uppercase tracking-wide">Configuration Differences</h2>
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-800">
                    <th className="text-left py-2 text-gray-500">Parameter</th>
                    <th className="text-left py-2" style={{ color: COMPARE_COLORS[0] }}>Session A</th>
                    <th className="text-left py-2" style={{ color: COMPARE_COLORS[1] }}>Session B</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(Object.values(comparison.config_diffs)[0] || {}).map(([param, values]) => (
                    <tr key={param} className="border-b border-gray-800/50">
                      <td className="py-1.5 text-gray-300 font-mono text-xs">{param}</td>
                      <td className="py-1.5 text-gray-200">{JSON.stringify(values[0])}</td>
                      <td className="py-1.5 text-gray-200">{JSON.stringify(values[1])}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* Overlaid Charts */}
          <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
            <OverlaidChart title="Population Size" data={overlaidData} keyA="pop_a" keyB="pop_b" labels={sessionNames} />
            <OverlaidChart title="Mean Contribution" data={overlaidData} keyA="contrib_a" keyB="contrib_b" labels={sessionNames} />
            <OverlaidChart title="Mean Suffering" data={overlaidData} keyA="suffer_a" keyB="suffer_b" labels={sessionNames} />
          </div>
        </>
      )}
    </div>
  );
}

function OverlaidChart({ title, data, keyA, keyB, labels }: {
  title: string; data: Record<string, unknown>[]; keyA: string; keyB: string; labels: string[];
}) {
  return (
    <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
      <h3 className="mb-2 text-sm font-semibold text-gray-400 uppercase tracking-wide">{title}</h3>
      <ResponsiveContainer width="100%" height={280}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis dataKey="generation" stroke="#6b7280" tick={{ fontSize: 11 }} />
          <YAxis stroke="#6b7280" tick={{ fontSize: 11 }} />
          <Tooltip contentStyle={{ backgroundColor: '#111827', border: '1px solid #374151', borderRadius: '8px' }} />
          <Line type="monotone" dataKey={keyA} stroke={COMPARE_COLORS[0]} strokeWidth={2} dot={false} name={labels[0]} connectNulls />
          <Line type="monotone" dataKey={keyB} stroke={COMPARE_COLORS[1]} strokeWidth={2} dot={false} name={labels[1]} connectNulls />
          <Legend wrapperStyle={{ fontSize: 11, color: '#9ca3af' }} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

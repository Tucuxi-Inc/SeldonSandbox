import { useState } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
} from 'recharts';
import { useSimulationStore } from '../../../stores/simulation';
import { EmptyState } from '../../shared/EmptyState';
import * as api from '../../../api/client';
import type { SensitivityReport } from '../../../types';
import { SENSITIVITY_COLORS } from '../../../lib/constants';

const TARGET_METRICS = [
  { value: 'mean_contribution', label: 'Mean Contribution' },
  { value: 'mean_suffering', label: 'Mean Suffering' },
  { value: 'breakthroughs', label: 'Total Breakthroughs' },
  { value: 'population_size', label: 'Final Population' },
  { value: 'trait_entropy', label: 'Trait Entropy' },
  { value: 'total_births', label: 'Total Births' },
  { value: 'total_deaths', label: 'Total Deaths' },
];

export function ParameterSensitivityView() {
  const { sessions } = useSimulationStore();
  const [selectedSessions, setSelectedSessions] = useState<string[]>([]);
  const [targetMetric, setTargetMetric] = useState('mean_contribution');
  const [report, setReport] = useState<SensitivityReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const completedSessions = sessions.filter((s) => s.status === 'completed' || s.current_generation > 0);

  const toggleSession = (id: string) => {
    setSelectedSessions((prev) =>
      prev.includes(id) ? prev.filter((s) => s !== id) : [...prev, id]
    );
  };

  const runAnalysis = async () => {
    if (selectedSessions.length < 2) {
      setError('Select at least 2 sessions');
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const data = await api.computeSensitivity(selectedSessions[0], selectedSessions, targetMetric);
      setReport(data);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Analysis failed');
    }
    setLoading(false);
  };

  if (completedSessions.length < 2) {
    return <EmptyState message="Complete at least 2 simulations to analyze parameter sensitivity" />;
  }

  // Tornado chart data
  const tornadoData = report?.tornado_data
    .slice(0, 15)
    .map((t) => ({
      parameter: t.parameter.length > 25 ? t.parameter.slice(0, 25) + '...' : t.parameter,
      fullParameter: t.parameter,
      positive: t.swing >= 0 ? t.swing : 0,
      negative: t.swing < 0 ? t.swing : 0,
      swing: t.swing,
    })) ?? [];

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-gray-100">Parameter Sensitivity</h1>

      {/* Controls */}
      <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
          {/* Session selector */}
          <div className="lg:col-span-2">
            <label className="mb-2 block text-sm text-gray-400">Select Sessions (min 2)</label>
            <div className="space-y-1 max-h-48 overflow-y-auto rounded border border-gray-800 bg-gray-950 p-2">
              {completedSessions.map((s) => (
                <label
                  key={s.id}
                  className="flex items-center gap-2 cursor-pointer rounded px-2 py-1 text-sm hover:bg-gray-900"
                >
                  <input
                    type="checkbox"
                    checked={selectedSessions.includes(s.id)}
                    onChange={() => toggleSession(s.id)}
                    className="accent-blue-600"
                  />
                  <span className="text-gray-200">{s.name}</span>
                  <span className="text-xs text-gray-500">({s.id}) — Gen {s.current_generation}</span>
                </label>
              ))}
            </div>
          </div>

          {/* Target metric + run */}
          <div className="space-y-3">
            <div>
              <label className="mb-1 block text-sm text-gray-400">Target Metric</label>
              <select
                className="w-full rounded-md border border-gray-700 bg-gray-800 px-3 py-2 text-sm text-gray-200"
                value={targetMetric}
                onChange={(e) => setTargetMetric(e.target.value)}
              >
                {TARGET_METRICS.map((m) => (
                  <option key={m.value} value={m.value}>{m.label}</option>
                ))}
              </select>
            </div>
            <button
              onClick={runAnalysis}
              disabled={selectedSessions.length < 2 || loading}
              className="w-full rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-40"
            >
              {loading ? 'Analyzing...' : 'Run Analysis'}
            </button>
            {error && <p className="text-sm text-red-400">{error}</p>}
          </div>
        </div>
      </div>

      {report && (
        <>
          {/* Tornado Chart */}
          {tornadoData.length > 0 && (
            <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
              <h2 className="mb-4 text-lg font-semibold text-gray-200">
                Tornado Chart — Impact on {TARGET_METRICS.find((m) => m.value === report.target_metric)?.label || report.target_metric}
              </h2>
              <ResponsiveContainer width="100%" height={Math.max(250, tornadoData.length * 35)}>
                <BarChart data={tornadoData} layout="vertical" margin={{ left: 120 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis type="number" stroke="#6b7280" fontSize={11} />
                  <YAxis type="category" dataKey="parameter" stroke="#6b7280" fontSize={11} width={120} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: '8px' }}
                    labelStyle={{ color: '#9ca3af' }}
                    formatter={(value: number, name: string) => [value.toFixed(4), name === 'positive' ? 'Positive Swing' : 'Negative Swing']}
                  />
                  <Bar dataKey="positive" fill={SENSITIVITY_COLORS.positive} radius={[0, 4, 4, 0]} />
                  <Bar dataKey="negative" fill={SENSITIVITY_COLORS.negative} radius={[4, 0, 0, 4]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Sensitivity ranking table */}
          <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
            <h2 className="mb-4 text-lg font-semibold text-gray-200">
              Sensitivity Ranking ({report.sensitivities.length} varying parameters)
            </h2>
            {report.sensitivities.length === 0 ? (
              <p className="text-sm text-gray-400">No varying parameters found between selected sessions.</p>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-gray-800 text-left text-gray-400">
                      <th className="px-3 py-2">Parameter</th>
                      <th className="px-3 py-2">Correlation</th>
                      <th className="px-3 py-2">Impact</th>
                      <th className="px-3 py-2">Value Range</th>
                      <th className="px-3 py-2">Outcome Range</th>
                    </tr>
                  </thead>
                  <tbody>
                    {report.sensitivities.map((s) => (
                      <tr key={s.parameter} className="border-b border-gray-800/50">
                        <td className="px-3 py-2 font-mono text-gray-200">{s.parameter}</td>
                        <td className="px-3 py-2">
                          <span className={s.correlation >= 0 ? 'text-green-400' : 'text-red-400'}>
                            {s.correlation >= 0 ? '+' : ''}{s.correlation.toFixed(3)}
                          </span>
                        </td>
                        <td className="px-3 py-2">
                          <div className="flex items-center gap-2">
                            <div className="h-1.5 w-20 rounded-full bg-gray-700">
                              <div
                                className="h-1.5 rounded-full bg-blue-500"
                                style={{
                                  width: `${Math.min(100, (s.impact / Math.max(...report.sensitivities.map((x) => x.impact), 0.001)) * 100)}%`,
                                }}
                              />
                            </div>
                            <span className="text-xs text-gray-400">{s.impact.toFixed(4)}</span>
                          </div>
                        </td>
                        <td className="px-3 py-2 font-mono text-xs text-gray-400">
                          {typeof s.min_value === 'number' ? s.min_value.toFixed(3) : String(s.min_value)}
                          {' → '}
                          {typeof s.max_value === 'number' ? s.max_value.toFixed(3) : String(s.max_value)}
                        </td>
                        <td className="px-3 py-2 font-mono text-xs text-gray-400">
                          {s.min_outcome.toFixed(4)} → {s.max_outcome.toFixed(4)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}

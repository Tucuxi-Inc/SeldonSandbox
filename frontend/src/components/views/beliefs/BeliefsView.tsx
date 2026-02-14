import { useState, useEffect } from 'react';
import { useSimulationStore } from '../../../stores/simulation';
import * as api from '../../../api/client';
import type { BeliefOverview, EpistemologyDistribution, AccuracyByDomain } from '../../../types';
import { EPISTEMOLOGY_COLORS, BELIEF_DOMAIN_COLORS } from '../../../lib/constants';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
} from 'recharts';

export function BeliefsView() {
  const { activeSessionId } = useSimulationStore();
  const [overview, setOverview] = useState<BeliefOverview | null>(null);
  const [epistemology, setEpistemology] = useState<EpistemologyDistribution | null>(null);
  const [accuracy, setAccuracy] = useState<AccuracyByDomain | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!activeSessionId) return;
    setLoading(true);
    Promise.all([
      api.getBeliefsOverview(activeSessionId).catch(() => null),
      api.getEpistemologyDistribution(activeSessionId).catch(() => null),
      api.getAccuracyByDomain(activeSessionId).catch(() => null),
    ]).then(([o, e, a]) => {
      setOverview(o);
      setEpistemology(e);
      setAccuracy(a);
    }).finally(() => setLoading(false));
  }, [activeSessionId]);

  if (!activeSessionId) {
    return <div className="p-6 text-gray-400">Create a session to view beliefs.</div>;
  }

  if (loading) return <div className="p-6 text-gray-400">Loading...</div>;
  if (overview && !overview.enabled) {
    return <div className="p-6 text-gray-400">Enable epistemology in Mission Control to view belief systems.</div>;
  }

  const epiData = epistemology?.distribution
    ? Object.entries(epistemology.distribution).map(([type, info]) => ({
        type: type.charAt(0).toUpperCase() + type.slice(1),
        key: type,
        count: info.count,
        mean_accuracy: info.mean_accuracy,
      }))
    : [];

  const domainData = accuracy?.domains
    ? Object.entries(accuracy.domains).map(([domain, info]) => ({
        domain: domain.charAt(0).toUpperCase() + domain.slice(1),
        key: domain,
        mean_accuracy: Math.round(info.mean_accuracy * 100) / 100,
        count: info.count,
      }))
    : [];

  const societalBeliefs = overview?.societal_beliefs ?? [];

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-gray-100">Belief Systems</h1>

      {/* Summary Stats */}
      {overview && overview.enabled && (
        <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
          <StatCard label="Total Beliefs" value={overview.total_beliefs} />
          <StatCard label="Mean Conviction" value={overview.mean_conviction.toFixed(3)} />
          <StatCard label="Mean Accuracy" value={overview.mean_accuracy.toFixed(3)} />
          <StatCard label="Societal Beliefs" value={societalBeliefs.length} />
        </div>
      )}

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Epistemology Distribution */}
        {epiData.length > 0 && (
          <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
            <h2 className="mb-3 text-lg font-semibold text-gray-200">Epistemology Distribution</h2>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={epiData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="type" stroke="#9ca3af" tick={{ fontSize: 12 }} />
                <YAxis stroke="#9ca3af" tick={{ fontSize: 11 }} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }}
                  formatter={((value: any, name: string) => {
                    if (name === 'count') return [value, 'Count'];
                    return [Number(value).toFixed(3), 'Mean Accuracy'];
                  }) as any}
                />
                <Bar dataKey="count" name="count">
                  {epiData.map((entry) => (
                    <Cell key={entry.key} fill={EPISTEMOLOGY_COLORS[entry.key] ?? '#6B7280'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <div className="mt-2 flex flex-wrap gap-2">
              {epiData.map((d) => (
                <span key={d.key} className="flex items-center gap-1 text-xs text-gray-400">
                  <span className="inline-block h-2.5 w-2.5 rounded" style={{ backgroundColor: EPISTEMOLOGY_COLORS[d.key] }} />
                  {d.type}: accuracy {d.mean_accuracy.toFixed(2)}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Accuracy by Domain */}
        {domainData.length > 0 && (
          <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
            <h2 className="mb-3 text-lg font-semibold text-gray-200">Accuracy by Domain</h2>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={domainData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis type="number" stroke="#9ca3af" tick={{ fontSize: 11 }} domain={[0, 1]} />
                <YAxis type="category" dataKey="domain" stroke="#9ca3af" tick={{ fontSize: 11 }} width={90} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }}
                  formatter={((value: any) => [Number(value).toFixed(3), 'Accuracy']) as any}
                />
                <Bar dataKey="mean_accuracy" name="Accuracy">
                  {domainData.map((entry) => (
                    <Cell key={entry.key} fill={BELIEF_DOMAIN_COLORS[entry.key] ?? '#6B7280'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <div className="mt-2 flex flex-wrap gap-2">
              {domainData.map((d) => (
                <span key={d.key} className="text-xs text-gray-500">{d.domain}: {d.count} beliefs</span>
              ))}
            </div>
          </div>
        )}

        {/* Societal Beliefs Table */}
        {societalBeliefs.length > 0 && (
          <div className="rounded-lg border border-gray-800 bg-gray-900 p-4 lg:col-span-2">
            <h2 className="mb-3 text-lg font-semibold text-gray-200">Societal Beliefs</h2>
            <div className="max-h-72 overflow-y-auto">
              <table className="w-full text-sm">
                <thead className="sticky top-0 bg-gray-900">
                  <tr className="border-b border-gray-800 text-left text-gray-500">
                    <th className="pb-2 pr-4">Content</th>
                    <th className="pb-2 pr-4">Domain</th>
                    <th className="pb-2 pr-4">Epistemology</th>
                    <th className="pb-2 pr-4 text-right">Conviction</th>
                    <th className="pb-2 text-right">Accuracy</th>
                  </tr>
                </thead>
                <tbody>
                  {societalBeliefs.map((belief, i) => {
                    const content = (belief.content as string) ?? '—';
                    const domain = (belief.domain as string) ?? '—';
                    const epi = (belief.epistemology as string) ?? '—';
                    const conviction = typeof belief.conviction === 'number' ? belief.conviction.toFixed(2) : '—';
                    const acc = typeof belief.accuracy === 'number' ? belief.accuracy.toFixed(2) : '—';
                    return (
                      <tr key={i} className="border-b border-gray-800/50 text-gray-300">
                        <td className="max-w-xs truncate py-1.5 pr-4">{content}</td>
                        <td className="py-1.5 pr-4">
                          <span className="inline-block h-2 w-2 rounded-full mr-1" style={{ backgroundColor: BELIEF_DOMAIN_COLORS[domain] ?? '#6B7280' }} />
                          {domain}
                        </td>
                        <td className="py-1.5 pr-4">
                          <span className="inline-block h-2 w-2 rounded-full mr-1" style={{ backgroundColor: EPISTEMOLOGY_COLORS[epi] ?? '#6B7280' }} />
                          {epi}
                        </td>
                        <td className="py-1.5 pr-4 text-right font-mono">{conviction}</td>
                        <td className="py-1.5 text-right font-mono">{acc}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function StatCard({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
      <div className="text-sm text-gray-400">{label}</div>
      <div className="mt-1 text-2xl font-bold text-gray-100">{value}</div>
    </div>
  );
}

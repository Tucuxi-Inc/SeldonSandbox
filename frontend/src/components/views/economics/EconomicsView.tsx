import { useState, useEffect } from 'react';
import { useSimulationStore } from '../../../stores/simulation';
import * as api from '../../../api/client';
import type { EconomicsOverview, TradeRoute, WealthDistribution, OccupationBreakdown } from '../../../types';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell,
} from 'recharts';

const PIE_COLORS = ['#3b82f6', '#22c55e', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#84cc16'];

export function EconomicsView() {
  const { activeSessionId } = useSimulationStore();
  const [overview, setOverview] = useState<EconomicsOverview | null>(null);
  const [trades, setTrades] = useState<TradeRoute[]>([]);
  const [wealth, setWealth] = useState<WealthDistribution | null>(null);
  const [occupations, setOccupations] = useState<OccupationBreakdown | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!activeSessionId) return;
    setLoading(true);
    Promise.all([
      api.getEconomicsOverview(activeSessionId).catch(() => null),
      api.getTradeRoutes(activeSessionId).catch(() => ({ enabled: false, routes: [] })),
      api.getWealthDistribution(activeSessionId).catch(() => null),
      api.getOccupations(activeSessionId).catch(() => null),
    ]).then(([o, t, w, occ]) => {
      setOverview(o);
      setTrades(t?.routes ?? []);
      setWealth(w);
      setOccupations(occ);
    }).finally(() => setLoading(false));
  }, [activeSessionId]);

  if (!activeSessionId) {
    return <div className="p-6 text-gray-400">Create a session to view economics.</div>;
  }

  if (loading) return <div className="p-6 text-gray-400">Loading...</div>;
  if (!overview?.enabled) {
    return <div className="p-6 text-gray-400">Economics extension not enabled. Enable it in Mission Control.</div>;
  }

  const gdpData = Object.entries(overview.gdp_by_community ?? {}).map(([name, gdp]) => ({
    name,
    gdp: Math.round(gdp as number * 100) / 100,
  }));

  const occupationData = occupations?.occupations
    ? Object.entries(occupations.occupations).map(([name, count]) => ({ name, value: count }))
    : [];

  const wealthPercentiles = wealth?.percentiles
    ? Object.entries(wealth.percentiles).map(([pct, val]) => ({ percentile: pct, wealth: Math.round(val * 100) / 100 }))
    : [];

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-gray-100">Economics</h1>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        <div className="rounded-lg border border-gray-800 bg-gray-900 p-4 text-center">
          <div className="text-sm text-gray-500">Total Wealth</div>
          <div className="text-2xl font-bold text-gray-100">{overview.total_wealth?.toFixed(1) ?? 0}</div>
        </div>
        <div className="rounded-lg border border-gray-800 bg-gray-900 p-4 text-center">
          <div className="text-sm text-gray-500">Gini Coefficient</div>
          <div className="text-2xl font-bold text-amber-400">{overview.gini_coefficient?.toFixed(3) ?? 0}</div>
        </div>
        <div className="rounded-lg border border-gray-800 bg-gray-900 p-4 text-center">
          <div className="text-sm text-gray-500">Trade Routes</div>
          <div className="text-2xl font-bold text-gray-100">{trades.length}</div>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* GDP */}
        {gdpData.length > 0 && (
          <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
            <h2 className="mb-3 text-lg font-semibold text-gray-200">GDP by Community</h2>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={gdpData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="name" stroke="#9ca3af" tick={{ fontSize: 11 }} />
                <YAxis stroke="#9ca3af" tick={{ fontSize: 11 }} />
                <Tooltip contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }} />
                <Bar dataKey="gdp" fill="#3b82f6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Occupations */}
        {occupationData.length > 0 && (
          <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
            <h2 className="mb-3 text-lg font-semibold text-gray-200">Occupation Breakdown</h2>
            <ResponsiveContainer width="100%" height={250}>
              <PieChart>
                <Pie
                  data={occupationData}
                  dataKey="value"
                  nameKey="name"
                  cx="50%"
                  cy="50%"
                  outerRadius={90}
                  label={({ name, percent }: any) => `${name} ${(percent * 100).toFixed(0)}%`}
                >
                  {occupationData.map((_, i) => (
                    <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Wealth Distribution */}
        {wealthPercentiles.length > 0 && (
          <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
            <h2 className="mb-3 text-lg font-semibold text-gray-200">Wealth Distribution</h2>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={wealthPercentiles}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="percentile" stroke="#9ca3af" tick={{ fontSize: 11 }} />
                <YAxis stroke="#9ca3af" tick={{ fontSize: 11 }} />
                <Tooltip contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }} />
                <Bar dataKey="wealth" fill="#f59e0b" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
            {wealth && (
              <div className="mt-2 flex gap-4 text-xs text-gray-500">
                <span>Mean: {wealth.mean_wealth?.toFixed(2)}</span>
                <span>Median: {wealth.median_wealth?.toFixed(2)}</span>
              </div>
            )}
          </div>
        )}

        {/* Trade Routes */}
        {trades.length > 0 && (
          <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
            <h2 className="mb-3 text-lg font-semibold text-gray-200">Trade Routes</h2>
            <div className="max-h-60 overflow-y-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-800 text-left text-gray-500">
                    <th className="pb-2">From</th>
                    <th className="pb-2">To</th>
                    <th className="pb-2">Resource</th>
                    <th className="pb-2">Volume</th>
                  </tr>
                </thead>
                <tbody>
                  {trades.map((r, i) => (
                    <tr key={i} className="border-b border-gray-800/50">
                      <td className="py-1.5 text-gray-300">{r.from_community}</td>
                      <td className="py-1.5 text-gray-300">{r.to_community}</td>
                      <td className="py-1.5 text-gray-400">{r.resource}</td>
                      <td className="py-1.5 font-mono text-gray-300">{r.volume.toFixed(1)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

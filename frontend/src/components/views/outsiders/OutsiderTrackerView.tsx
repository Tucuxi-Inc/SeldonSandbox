import { useState, useEffect } from 'react';
import { useSimulationStore } from '../../../stores/simulation';
import * as api from '../../../api/client';
import type { AgentSummary, OutsiderImpact } from '../../../types';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';

const REGION_COLORS: Record<string, string> = {
  under_processing: '#6b7280',
  optimal: '#22c55e',
  deep: '#3b82f6',
  sacrificial: '#f59e0b',
  pathological: '#ef4444',
};

export function OutsiderTrackerView() {
  const { activeSessionId } = useSimulationStore();
  const [outsiders, setOutsiders] = useState<AgentSummary[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [impact, setImpact] = useState<OutsiderImpact | null>(null);
  const [ripple, setRipple] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!activeSessionId) return;
    setLoading(true);
    Promise.all([
      api.getOutsiders(activeSessionId),
      api.getRippleReport(activeSessionId),
    ]).then(([o, r]) => {
      setOutsiders(o);
      setRipple(r);
      if (o.length > 0 && !selectedId) setSelectedId(o[0].id);
    }).catch(() => {}).finally(() => setLoading(false));
  }, [activeSessionId]);

  useEffect(() => {
    if (!activeSessionId || !selectedId) { setImpact(null); return; }
    api.getOutsiderImpact(activeSessionId, selectedId)
      .then(setImpact).catch(() => setImpact(null));
  }, [activeSessionId, selectedId]);

  if (!activeSessionId) {
    return <div className="p-6 text-gray-400">Create a session to track outsiders.</div>;
  }

  const rippleTimeline = ripple?.outsider_fraction_over_time as number[] ?? [];
  const rippleGens = rippleTimeline.map((_, i) => i);
  const rippleData = rippleGens.map((g, i) => ({ generation: g, fraction: rippleTimeline[i] ?? 0 }));

  const injectionGens = outsiders
    .map((o) => (o as unknown as { injection_generation: number | null }).injection_generation)
    .filter((g): g is number => g !== null);

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-gray-100">Outsider Tracker</h1>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        {/* Outsider Registry */}
        <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
          <h2 className="mb-3 text-lg font-semibold text-gray-200">Outsider Registry</h2>
          {loading && <div className="text-sm text-gray-500">Loading...</div>}
          {!loading && outsiders.length === 0 && (
            <div className="text-sm text-gray-500">No outsiders injected yet.</div>
          )}
          <div className="space-y-2 max-h-[60vh] overflow-y-auto">
            {outsiders.map((o) => (
              <button
                key={o.id}
                onClick={() => setSelectedId(o.id)}
                className={`w-full rounded border p-2 text-left transition-colors ${
                  selectedId === o.id
                    ? 'border-blue-600 bg-gray-800'
                    : 'border-gray-800 bg-gray-950 hover:bg-gray-800'
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-gray-200">{o.name}</span>
                  <span
                    className="rounded px-1.5 py-0.5 text-xs font-medium"
                    style={{ backgroundColor: REGION_COLORS[o.processing_region] ?? '#6b7280', color: '#fff' }}
                  >
                    {o.processing_region.replace('_', ' ')}
                  </span>
                </div>
                <div className="mt-1 flex items-center gap-2 text-xs text-gray-500">
                  <span>{o.is_alive ? 'Alive' : 'Dead'}</span>
                  <span>Gen {o.generation}</span>
                  <span>Age {o.age}</span>
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Right panels */}
        <div className="lg:col-span-2 space-y-4">
          {/* Ripple Impact */}
          <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
            <h2 className="mb-3 text-lg font-semibold text-gray-200">Outsider Fraction Over Time</h2>
            {rippleData.length > 0 ? (
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={rippleData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="generation" stroke="#9ca3af" tick={{ fontSize: 12 }} />
                  <YAxis stroke="#9ca3af" tick={{ fontSize: 12 }} domain={[0, 'auto']} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }}
                    labelStyle={{ color: '#9ca3af' }}
                    formatter={((value: any) => [Number(value).toFixed(4), 'Fraction']) as any}
                  />
                  {injectionGens.map((g, i) => (
                    <ReferenceLine key={i} x={g} stroke="#f59e0b" strokeDasharray="3 3" />
                  ))}
                  <Line type="monotone" dataKey="fraction" stroke="#3b82f6" dot={false} strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="text-sm text-gray-500">No ripple data available.</div>
            )}
          </div>

          {/* Selected Outsider Detail */}
          {impact && (
            <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
              <h2 className="mb-3 text-lg font-semibold text-gray-200">
                {impact.agent.name} — Detail
              </h2>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <div className="text-gray-500">Origin</div>
                  <div className="text-gray-200">{impact.outsider_origin ?? 'Unknown'}</div>
                </div>
                <div>
                  <div className="text-gray-500">Injected at Generation</div>
                  <div className="text-gray-200">{impact.injection_generation ?? 'N/A'}</div>
                </div>
                <div>
                  <div className="text-gray-500">Gender</div>
                  <div className="text-gray-200">{impact.gender ?? 'Unspecified'}</div>
                </div>
                <div>
                  <div className="text-gray-500">Trait Distance from Mean</div>
                  <div className="font-mono text-gray-200">{impact.trait_distance_from_mean}</div>
                </div>
                <div>
                  <div className="text-gray-500">Descendant Count</div>
                  <div className="text-2xl font-bold text-gray-100">{impact.descendant_count}</div>
                </div>
                <div>
                  <div className="text-gray-500">Status</div>
                  <div className={impact.agent.is_alive ? 'text-green-400' : 'text-red-400'}>
                    {impact.agent.is_alive ? 'Alive' : 'Dead'}
                  </div>
                </div>
              </div>

              {impact.descendants.length > 0 && (
                <div className="mt-4">
                  <h3 className="mb-2 text-sm font-medium text-gray-400">Descendants</h3>
                  <div className="max-h-40 overflow-y-auto space-y-1">
                    {impact.descendants.map((d) => (
                      <div key={d.id} className="flex items-center justify-between rounded bg-gray-950 px-2 py-1 text-xs">
                        <span className="text-gray-300">{d.name}</span>
                        <span className="text-gray-500">Gen {d.generation} | Age {d.age}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

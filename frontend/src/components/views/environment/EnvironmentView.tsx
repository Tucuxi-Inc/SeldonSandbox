import { useState, useEffect } from 'react';
import { useSimulationStore } from '../../../stores/simulation';
import * as api from '../../../api/client';
import type { ClimateState, EnvironmentEvent, DiseaseInfo } from '../../../types';

const SEASON_ICONS: Record<string, string> = {
  spring: 'Spring',
  summer: 'Summer',
  autumn: 'Autumn',
  winter: 'Winter',
};

const SEVERITY_COLORS: Record<string, string> = {
  low: 'bg-yellow-900 text-yellow-300',
  medium: 'bg-orange-900 text-orange-300',
  high: 'bg-red-900 text-red-300',
  critical: 'bg-red-700 text-red-100',
};

export function EnvironmentView() {
  const { activeSessionId } = useSimulationStore();
  const [climate, setClimate] = useState<ClimateState | null>(null);
  const [events, setEvents] = useState<EnvironmentEvent[]>([]);
  const [diseases, setDiseases] = useState<DiseaseInfo[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!activeSessionId) return;
    setLoading(true);
    Promise.all([
      api.getClimate(activeSessionId).catch(() => null),
      api.getEnvironmentEvents(activeSessionId).catch(() => ({ enabled: false, events: [] })),
      api.getDiseases(activeSessionId).catch(() => ({ enabled: false, diseases: [] })),
    ]).then(([c, e, d]) => {
      setClimate(c);
      setEvents(e?.events ?? []);
      setDiseases(d?.diseases ?? []);
    }).finally(() => setLoading(false));
  }, [activeSessionId]);

  if (!activeSessionId) {
    return <div className="p-6 text-gray-400">Create a session to view environment.</div>;
  }

  if (loading) return <div className="p-6 text-gray-400">Loading...</div>;
  if (!climate?.enabled) {
    return <div className="p-6 text-gray-400">Environment extension not enabled. Enable it in Mission Control.</div>;
  }

  const locations = climate.locations ?? {};

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-gray-100">Environment</h1>

      {/* Season indicator */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        <div className="rounded-lg border border-gray-800 bg-gray-900 p-4 text-center">
          <div className="text-sm text-gray-500">Current Season</div>
          <div className="text-3xl font-bold text-gray-100">
            {SEASON_ICONS[climate.current_season] ?? climate.current_season}
          </div>
          <div className="mt-1 h-2 rounded-full bg-gray-800">
            <div
              className="h-2 rounded-full bg-green-600 transition-all"
              style={{ width: `${(climate.season_progress ?? 0) * 100}%` }}
            />
          </div>
        </div>
        <div className="rounded-lg border border-gray-800 bg-gray-900 p-4 text-center">
          <div className="text-sm text-gray-500">Total Events</div>
          <div className="text-2xl font-bold text-gray-100">{events.length}</div>
        </div>
        <div className="rounded-lg border border-gray-800 bg-gray-900 p-4 text-center">
          <div className="text-sm text-gray-500">Active Diseases</div>
          <div className="text-2xl font-bold text-red-400">{diseases.filter((d) => d.active).length}</div>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Climate State */}
        {Object.keys(locations).length > 0 && (
          <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
            <h2 className="mb-3 text-lg font-semibold text-gray-200">Climate by Location</h2>
            <div className="space-y-3">
              {Object.entries(locations).map(([locId, state]) => (
                <div key={locId} className="rounded border border-gray-800 bg-gray-950 p-3">
                  <div className="text-sm font-medium text-gray-300">{locId}</div>
                  <div className="mt-2 grid grid-cols-2 gap-2">
                    <div>
                      <div className="text-xs text-gray-500">Temperature</div>
                      <div className="flex items-center gap-2">
                        <div className="h-2 flex-1 rounded-full bg-gray-800">
                          <div
                            className="h-2 rounded-full bg-orange-500"
                            style={{ width: `${Math.min(100, Math.max(0, (state.temperature + 10) / 60 * 100))}%` }}
                          />
                        </div>
                        <span className="font-mono text-xs text-gray-400">{state.temperature?.toFixed(1)}</span>
                      </div>
                    </div>
                    <div>
                      <div className="text-xs text-gray-500">Rainfall</div>
                      <div className="flex items-center gap-2">
                        <div className="h-2 flex-1 rounded-full bg-gray-800">
                          <div
                            className="h-2 rounded-full bg-blue-500"
                            style={{ width: `${Math.min(100, (state.rainfall ?? 0) * 100)}%` }}
                          />
                        </div>
                        <span className="font-mono text-xs text-gray-400">{state.rainfall?.toFixed(2)}</span>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Events Timeline */}
        <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
          <h2 className="mb-3 text-lg font-semibold text-gray-200">Event Timeline</h2>
          {events.length === 0 ? (
            <div className="text-sm text-gray-500">No events recorded.</div>
          ) : (
            <div className="max-h-80 overflow-y-auto space-y-2">
              {events.slice().reverse().map((e, i) => (
                <div key={i} className="flex gap-3 rounded border border-gray-800 bg-gray-950 p-2">
                  <div className="text-xs text-gray-500 whitespace-nowrap">Gen {e.generation}</div>
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <span className="text-sm text-gray-200">{e.type}</span>
                      <span className={`rounded px-1.5 py-0.5 text-xs ${SEVERITY_COLORS[e.severity] ?? 'bg-gray-800 text-gray-400'}`}>
                        {e.severity}
                      </span>
                    </div>
                    <div className="text-xs text-gray-500">{e.description}</div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Disease Tracker */}
        {diseases.length > 0 && (
          <div className="rounded-lg border border-gray-800 bg-gray-900 p-4 lg:col-span-2">
            <h2 className="mb-3 text-lg font-semibold text-gray-200">Disease Tracker</h2>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-800 text-left text-gray-500">
                    <th className="pb-2">Disease</th>
                    <th className="pb-2">Status</th>
                    <th className="pb-2">Infections</th>
                    <th className="pb-2">Mortality Rate</th>
                  </tr>
                </thead>
                <tbody>
                  {diseases.map((d, i) => (
                    <tr key={i} className="border-b border-gray-800/50">
                      <td className="py-2 text-gray-300">{d.name}</td>
                      <td className="py-2">
                        <span className={`rounded px-1.5 py-0.5 text-xs ${d.active ? 'bg-red-900 text-red-300' : 'bg-green-900 text-green-300'}`}>
                          {d.active ? 'Active' : 'Contained'}
                        </span>
                      </td>
                      <td className="py-2 font-mono text-gray-300">{d.infection_count}</td>
                      <td className="py-2 font-mono text-gray-300">{(d.mortality_rate * 100).toFixed(1)}%</td>
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

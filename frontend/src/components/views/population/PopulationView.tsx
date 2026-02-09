import { useSimulationStore } from '../../../stores/simulation';
import { EmptyState } from '../../shared/EmptyState';
import { REGION_COLORS, REGION_ORDER, REGION_LABELS } from '../../../lib/constants';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ComposedChart, Line, Bar,
} from 'recharts';

export function PopulationView() {
  const { metrics } = useSimulationStore();

  if (metrics.length === 0) {
    return <EmptyState message="Run a simulation to see population data" />;
  }

  // Region stacked area data
  const regionData = metrics.map((m) => ({
    generation: m.generation,
    ...Object.fromEntries(REGION_ORDER.map((r) => [r, m.region_fractions[r] || 0])),
  }));

  // Population timeline data
  const popData = metrics.map((m) => ({
    generation: m.generation,
    population: m.population_size,
    births: m.births,
    deaths: -m.deaths,
  }));

  // Trait heatmap data
  const traitNames = metrics.length > 0 ? Object.keys(metrics[0].trait_means) : [];

  // Breakthrough timeline data
  const breakthroughData = metrics.map((m) => ({
    generation: m.generation,
    breakthroughs: m.breakthroughs,
    contribution: m.mean_contribution,
  }));

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-gray-100">Population Overview</h1>

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
        {/* Region Distribution */}
        <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
          <h2 className="mb-4 text-sm font-semibold text-gray-400 uppercase tracking-wide">
            Processing Region Distribution
          </h2>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={regionData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
              <XAxis dataKey="generation" stroke="#6b7280" tick={{ fontSize: 12 }} />
              <YAxis stroke="#6b7280" tick={{ fontSize: 12 }} domain={[0, 1]} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
              <Tooltip
                contentStyle={{ backgroundColor: '#111827', border: '1px solid #374151', borderRadius: '8px' }}
                labelStyle={{ color: '#9ca3af' }}
              />
              {REGION_ORDER.map((region) => (
                <Area
                  key={region}
                  type="monotone"
                  dataKey={region}
                  stackId="1"
                  fill={REGION_COLORS[region]}
                  stroke={REGION_COLORS[region]}
                  fillOpacity={0.8}
                  name={REGION_LABELS[region]}
                />
              ))}
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Population Timeline */}
        <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
          <h2 className="mb-4 text-sm font-semibold text-gray-400 uppercase tracking-wide">
            Population Size
          </h2>
          <ResponsiveContainer width="100%" height={300}>
            <ComposedChart data={popData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
              <XAxis dataKey="generation" stroke="#6b7280" tick={{ fontSize: 12 }} />
              <YAxis stroke="#6b7280" tick={{ fontSize: 12 }} />
              <Tooltip
                contentStyle={{ backgroundColor: '#111827', border: '1px solid #374151', borderRadius: '8px' }}
                labelStyle={{ color: '#9ca3af' }}
              />
              <Bar dataKey="births" fill="#10B981" opacity={0.6} name="Births" />
              <Bar dataKey="deaths" fill="#EF4444" opacity={0.6} name="Deaths" />
              <Line type="monotone" dataKey="population" stroke="#3B82F6" strokeWidth={2} dot={false} name="Population" />
            </ComposedChart>
          </ResponsiveContainer>
        </div>

        {/* Trait Drift Heatmap */}
        <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
          <h2 className="mb-4 text-sm font-semibold text-gray-400 uppercase tracking-wide">
            Trait Means Over Time
          </h2>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr>
                  <th className="text-left text-gray-500 py-1 px-1 sticky left-0 bg-gray-900">Trait</th>
                  {metrics.map((m) => (
                    <th key={m.generation} className="text-center text-gray-600 px-0.5 min-w-[20px]">
                      {m.generation}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {traitNames.map((trait) => (
                  <tr key={trait}>
                    <td className="text-gray-400 py-0.5 px-1 whitespace-nowrap sticky left-0 bg-gray-900">{trait}</td>
                    {metrics.map((m) => {
                      const val = m.trait_means[trait] || 0;
                      const hue = val * 120; // 0=red, 60=yellow, 120=green
                      return (
                        <td
                          key={m.generation}
                          className="px-0.5"
                          title={`${trait}: ${val.toFixed(3)} (gen ${m.generation})`}
                        >
                          <div
                            className="h-4 w-full rounded-sm"
                            style={{ backgroundColor: `hsl(${hue}, 70%, 40%)` }}
                          />
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Breakthrough Timeline */}
        <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
          <h2 className="mb-4 text-sm font-semibold text-gray-400 uppercase tracking-wide">
            Breakthroughs & Contribution
          </h2>
          <ResponsiveContainer width="100%" height={300}>
            <ComposedChart data={breakthroughData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
              <XAxis dataKey="generation" stroke="#6b7280" tick={{ fontSize: 12 }} />
              <YAxis yAxisId="left" stroke="#6b7280" tick={{ fontSize: 12 }} />
              <YAxis yAxisId="right" orientation="right" stroke="#6b7280" tick={{ fontSize: 12 }} />
              <Tooltip
                contentStyle={{ backgroundColor: '#111827', border: '1px solid #374151', borderRadius: '8px' }}
                labelStyle={{ color: '#9ca3af' }}
              />
              <Bar yAxisId="left" dataKey="breakthroughs" fill="#F59E0B" opacity={0.8} name="Breakthroughs" />
              <Line yAxisId="right" type="monotone" dataKey="contribution" stroke="#10B981" strokeWidth={2} dot={false} name="Mean Contribution" />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

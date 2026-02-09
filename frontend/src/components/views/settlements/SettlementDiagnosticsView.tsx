import { useState, useEffect, useRef } from 'react';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from 'recharts';
import { useSimulationStore } from '../../../stores/simulation';
import { EmptyState } from '../../shared/EmptyState';
import * as api from '../../../api/client';
import * as d3 from 'd3';
import type {
  SettlementOverview, ViabilityAssessment, MigrationHistory,
} from '../../../types';
import { REGION_COLORS, REGION_LABELS, REGION_ORDER } from '../../../lib/constants';

export function SettlementDiagnosticsView() {
  const { activeSessionId, metrics } = useSimulationStore();
  const svgRef = useRef<SVGSVGElement>(null);
  const [overview, setOverview] = useState<SettlementOverview | null>(null);
  const [migrationHistory, setMigrationHistory] = useState<MigrationHistory | null>(null);
  const [viability, setViability] = useState<Record<string, ViabilityAssessment>>({});
  const [selectedSettlement, setSelectedSettlement] = useState<string | null>(null);

  useEffect(() => {
    if (!activeSessionId) return;
    api.getSettlementsOverview(activeSessionId).then(setOverview).catch(() => {});
    api.getMigrationHistory(activeSessionId).then(setMigrationHistory).catch(() => {});
  }, [activeSessionId, metrics.length]);

  // Fetch viability for each settlement
  useEffect(() => {
    if (!activeSessionId || !overview?.enabled) return;
    const fetches = overview.settlements.map(async (s) => {
      try {
        const v = await api.getSettlementViability(activeSessionId, s.id);
        return [s.id, v] as const;
      } catch {
        return null;
      }
    });
    Promise.all(fetches).then((results) => {
      const map: Record<string, ViabilityAssessment> = {};
      for (const r of results) {
        if (r) map[r[0]] = r[1];
      }
      setViability(map);
    });
  }, [activeSessionId, overview]);

  // Draw settlement map
  useEffect(() => {
    if (!svgRef.current || !overview?.enabled || overview.settlements.length === 0) return;

    const svg = d3.select(svgRef.current);
    const width = svgRef.current.clientWidth;
    const height = 350;
    const margin = { top: 20, right: 30, bottom: 30, left: 40 };

    svg.selectAll('*').remove();
    svg.attr('width', width).attr('height', height);

    const settlements = overview.settlements;
    const xs = settlements.map((s) => s.coordinates[0]);
    const ys = settlements.map((s) => s.coordinates[1]);

    const xScale = d3.scaleLinear()
      .domain([Math.min(...xs) - 1, Math.max(...xs) + 1])
      .range([margin.left, width - margin.right]);

    const yScale = d3.scaleLinear()
      .domain([Math.min(...ys) - 1, Math.max(...ys) + 1])
      .range([height - margin.bottom, margin.top]);

    const sizeScale = d3.scaleSqrt()
      .domain([0, Math.max(...settlements.map((s) => s.population), 1)])
      .range([8, 40]);

    // Grid
    svg.append('g')
      .attr('transform', `translate(0, ${height - margin.bottom})`)
      .call(d3.axisBottom(xScale).ticks(8))
      .call((g) => g.selectAll('text').attr('fill', '#6b7280').style('font-size', '10px'))
      .call((g) => g.selectAll('line').attr('stroke', '#374151'))
      .call((g) => g.select('.domain').attr('stroke', '#374151'));

    svg.append('g')
      .attr('transform', `translate(${margin.left}, 0)`)
      .call(d3.axisLeft(yScale).ticks(8))
      .call((g) => g.selectAll('text').attr('fill', '#6b7280').style('font-size', '10px'))
      .call((g) => g.selectAll('line').attr('stroke', '#374151'))
      .call((g) => g.select('.domain').attr('stroke', '#374151'));

    function occupancyColor(ratio: number): string {
      if (ratio > 0.9) return '#EF4444';
      if (ratio > 0.7) return '#F59E0B';
      return '#10B981';
    }

    // Settlement circles
    svg.selectAll('circle.settlement')
      .data(settlements)
      .join('circle')
      .attr('class', 'settlement')
      .attr('cx', (d) => xScale(d.coordinates[0]))
      .attr('cy', (d) => yScale(d.coordinates[1]))
      .attr('r', (d) => sizeScale(d.population))
      .attr('fill', (d) => occupancyColor(d.occupancy_ratio))
      .attr('fill-opacity', 0.6)
      .attr('stroke', (d) => d.id === selectedSettlement ? '#fff' : occupancyColor(d.occupancy_ratio))
      .attr('stroke-width', (d) => d.id === selectedSettlement ? 3 : 1.5)
      .attr('cursor', 'pointer')
      .on('click', (_event, d) => setSelectedSettlement(d.id))
      .append('title')
      .text((d) => `${d.name}\nPop: ${d.population}/${d.carrying_capacity}\nOccupancy: ${(d.occupancy_ratio * 100).toFixed(0)}%`);

    // Labels
    svg.selectAll('text.label')
      .data(settlements)
      .join('text')
      .attr('class', 'label')
      .attr('x', (d) => xScale(d.coordinates[0]))
      .attr('y', (d) => yScale(d.coordinates[1]) - sizeScale(d.population) - 5)
      .attr('text-anchor', 'middle')
      .attr('fill', '#9ca3af')
      .style('font-size', '10px')
      .text((d) => d.name);
  }, [overview, selectedSettlement]);

  if (!activeSessionId || metrics.length === 0) {
    return <EmptyState message="Run a simulation to see settlement data" />;
  }

  if (!overview) {
    return <EmptyState message="Loading settlement data..." />;
  }

  if (!overview.enabled) {
    return <EmptyState message="Enable geography extension to see settlement data" />;
  }

  // Migration timeline data
  const timelineData = migrationHistory?.enabled
    ? migrationHistory.timeline.settlement_count_by_gen.map((count, i) => ({
        generation: i,
        settlements: count,
        migrations: migrationHistory.timeline.migrations_by_gen[i] || 0,
      }))
    : [];

  const selectedData = selectedSettlement
    ? overview.settlements.find((s) => s.id === selectedSettlement)
    : null;
  const selectedViability = selectedSettlement ? viability[selectedSettlement] : null;

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-gray-100">Settlement Diagnostics</h1>
      <p className="text-sm text-gray-400">
        {overview.settlements.length} settlements, {overview.total_population} total population,{' '}
        {overview.total_capacity} total capacity
      </p>

      {/* Settlement Map */}
      <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
        <h2 className="mb-4 text-lg font-semibold text-gray-200">Settlement Map</h2>
        <div className="mb-2 flex gap-4 text-xs text-gray-400">
          <span><span className="inline-block h-2 w-2 rounded-full bg-green-500 mr-1" />{'<70% capacity'}</span>
          <span><span className="inline-block h-2 w-2 rounded-full bg-yellow-500 mr-1" />70-90%</span>
          <span><span className="inline-block h-2 w-2 rounded-full bg-red-500 mr-1" />{'>90%'}</span>
        </div>
        <svg ref={svgRef} className="w-full" style={{ minHeight: 350 }} />
      </div>

      {/* Settlement Cards */}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
        {overview.settlements.map((s) => {
          const v = viability[s.id];
          return (
            <div
              key={s.id}
              onClick={() => setSelectedSettlement(s.id)}
              className={`cursor-pointer rounded-lg border p-4 transition-colors ${
                selectedSettlement === s.id
                  ? 'border-blue-600 bg-gray-800'
                  : 'border-gray-800 bg-gray-900 hover:border-gray-700'
              }`}
            >
              <div className="flex items-center justify-between">
                <h3 className="font-semibold text-gray-200">{s.name}</h3>
                {v && (
                  <span className={`rounded px-2 py-0.5 text-xs font-medium ${
                    v.viability_score >= 0.7 ? 'bg-green-900 text-green-200'
                    : v.viability_score >= 0.4 ? 'bg-yellow-900 text-yellow-200'
                    : 'bg-red-900 text-red-200'
                  }`}>
                    {(v.viability_score * 100).toFixed(0)}% viable
                  </span>
                )}
              </div>

              {/* Population bar */}
              <div className="mt-2">
                <div className="flex justify-between text-xs text-gray-400">
                  <span>Population</span>
                  <span>{s.population}/{s.carrying_capacity}</span>
                </div>
                <div className="mt-1 h-2 rounded-full bg-gray-700">
                  <div
                    className="h-2 rounded-full transition-all"
                    style={{
                      width: `${Math.min(s.occupancy_ratio * 100, 100)}%`,
                      backgroundColor: s.occupancy_ratio > 0.9 ? '#EF4444' : s.occupancy_ratio > 0.7 ? '#F59E0B' : '#10B981',
                    }}
                  />
                </div>
              </div>

              {/* Region breakdown mini-bar */}
              <div className="mt-3">
                <div className="text-xs text-gray-400 mb-1">Region Mix</div>
                <div className="flex h-2 rounded-full overflow-hidden bg-gray-700">
                  {REGION_ORDER.map((r) => {
                    const count = s.region_counts[r] || 0;
                    const pct = s.population > 0 ? (count / s.population) * 100 : 0;
                    if (pct === 0) return null;
                    return (
                      <div
                        key={r}
                        style={{ width: `${pct}%`, backgroundColor: REGION_COLORS[r] }}
                        title={`${REGION_LABELS[r]}: ${count}`}
                      />
                    );
                  })}
                </div>
              </div>

              {/* Risk factors */}
              {v && v.risk_factors.length > 0 && (
                <div className="mt-2 flex flex-wrap gap-1">
                  {v.risk_factors.map((rf) => (
                    <span key={rf} className="rounded bg-red-900/50 px-1.5 py-0.5 text-xs text-red-300">
                      {rf.replace(/_/g, ' ')}
                    </span>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Selected settlement composition */}
      {selectedData && selectedViability && (
        <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
          <h2 className="text-lg font-semibold text-gray-200">{selectedData.name} — Details</h2>
          <div className="mt-3 grid grid-cols-4 gap-4 text-sm">
            <div><span className="text-gray-500">Viability:</span> <span className="text-gray-200">{(selectedViability.viability_score * 100).toFixed(0)}%</span></div>
            <div><span className="text-gray-500">Checks:</span> <span className="text-gray-200">{selectedViability.checks_passed}/{selectedViability.checks_total}</span></div>
            <div><span className="text-gray-500">Group Size:</span> <span className="text-gray-200">{selectedViability.group_size}</span></div>
            <div><span className="text-gray-500">Resources:</span> <span className="text-gray-200">{selectedData.resource_richness.toFixed(2)}</span></div>
          </div>
        </div>
      )}

      {/* Migration Timeline */}
      {timelineData.length > 0 && (
        <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
          <h2 className="mb-4 text-lg font-semibold text-gray-200">Migration Timeline</h2>
          <ResponsiveContainer width="100%" height={250}>
            <AreaChart data={timelineData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="generation" stroke="#6b7280" fontSize={11} />
              <YAxis stroke="#6b7280" fontSize={11} />
              <Tooltip
                contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: '8px' }}
                labelStyle={{ color: '#9ca3af' }}
              />
              <Legend />
              <Area type="monotone" dataKey="settlements" stroke="#3B82F6" fill="#3B82F6" fillOpacity={0.3} name="Settlements" />
              <Area type="monotone" dataKey="migrations" stroke="#F59E0B" fill="#F59E0B" fillOpacity={0.3} name="Migrations" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Migration events table */}
      {migrationHistory?.enabled && migrationHistory.events.length > 0 && (
        <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
          <h2 className="mb-4 text-lg font-semibold text-gray-200">Migration Events</h2>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-800 text-left text-gray-400">
                  <th className="px-3 py-2">Gen</th>
                  <th className="px-3 py-2">Type</th>
                  <th className="px-3 py-2">Details</th>
                  <th className="px-3 py-2">Viability</th>
                  <th className="px-3 py-2">Risks</th>
                </tr>
              </thead>
              <tbody>
                {migrationHistory.events.map((evt, i) => (
                  <tr key={i} className="border-b border-gray-800/50">
                    <td className="px-3 py-2 text-gray-300">{evt.generation}</td>
                    <td className="px-3 py-2">
                      <span className={`rounded px-2 py-0.5 text-xs ${
                        evt.type === 'settlement_founded' ? 'bg-green-900 text-green-200'
                        : evt.type === 'migration_failed' ? 'bg-red-900 text-red-200'
                        : 'bg-gray-800 text-gray-300'
                      }`}>
                        {evt.type.replace(/_/g, ' ')}
                      </span>
                    </td>
                    <td className="px-3 py-2 text-gray-400">
                      {evt.founders ? `${evt.founders} founders` : ''}
                      {evt.count ? `${evt.count} relocated` : ''}
                      {evt.location_id ? ` → ${evt.location_id}` : ''}
                    </td>
                    <td className="px-3 py-2 text-gray-300">
                      {evt.viability != null ? `${(evt.viability * 100).toFixed(0)}%` : '-'}
                    </td>
                    <td className="px-3 py-2">
                      <div className="flex flex-wrap gap-1">
                        {(evt.risks || []).map((r: string) => (
                          <span key={r} className="rounded bg-gray-800 px-1 py-0.5 text-xs text-gray-400">
                            {r.replace(/_/g, ' ')}
                          </span>
                        ))}
                      </div>
                    </td>
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

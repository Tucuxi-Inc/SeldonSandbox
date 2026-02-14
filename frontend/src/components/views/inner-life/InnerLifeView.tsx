import { useState, useEffect, useRef } from 'react';
import { useSimulationStore } from '../../../stores/simulation';
import * as api from '../../../api/client';
import type { InnerLifeOverview, PQDistribution, MoodMapResponse, ExperientialDriftResponse } from '../../../types';
import { EXPERIENCE_DIM_LABELS, PQ_BUCKET_COLORS } from '../../../lib/constants';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  PieChart, Pie,
} from 'recharts';
import * as d3 from 'd3';

const EVENT_COLORS: Record<string, string> = {
  breakthrough: '#22C55E',
  deep_suffering: '#EF4444',
  pair_formed: '#EC4899',
  pair_dissolved: '#F97316',
  child_born: '#3B82F6',
  bereavement: '#6B7280',
  status_change: '#F59E0B',
  migration: '#8B5CF6',
  routine: '#9CA3AF',
};

export function InnerLifeView() {
  const { activeSessionId } = useSimulationStore();
  const [overview, setOverview] = useState<InnerLifeOverview | null>(null);
  const [pqDist, setPqDist] = useState<PQDistribution | null>(null);
  const [moodMap, setMoodMap] = useState<MoodMapResponse | null>(null);
  const [drift, setDrift] = useState<ExperientialDriftResponse | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!activeSessionId) return;
    setLoading(true);
    Promise.all([
      api.getInnerLifeOverview(activeSessionId).catch(() => null),
      api.getPQDistribution(activeSessionId).catch(() => null),
      api.getMoodMap(activeSessionId).catch(() => null),
      api.getExperientialDrift(activeSessionId).catch(() => null),
    ]).then(([o, p, m, d]) => {
      setOverview(o);
      setPqDist(p);
      setMoodMap(m);
      setDrift(d);
    }).finally(() => setLoading(false));
  }, [activeSessionId]);

  if (!activeSessionId) {
    return <div className="p-6 text-gray-400">Create a session to view inner life.</div>;
  }

  if (loading) return <div className="p-6 text-gray-400">Loading...</div>;
  if (overview && !overview.enabled) {
    return <div className="p-6 text-gray-400">Enable inner_life in Mission Control to view experiential mind data.</div>;
  }

  // PQ Distribution data
  const pqData = pqDist?.distribution
    ? Object.entries(pqDist.distribution).map(([bucket, count]) => ({
        bucket,
        count,
      }))
    : [];

  // Mood radar data
  const moodDims = ['valence', 'arousal', 'social_quality', 'agency', 'novelty', 'meaning'];
  const radarData = overview?.population_mood
    ? moodDims.map((dim, i) => ({
        dimension: EXPERIENCE_DIM_LABELS[dim] ?? dim,
        value: Math.max(0, overview.population_mood[i] ?? 0),
      }))
    : [];

  // Event type pie data
  const eventData = overview?.event_type_counts
    ? Object.entries(overview.event_type_counts)
        .filter(([, count]) => count > 0)
        .map(([type, count]) => ({
          name: type.replace(/_/g, ' '),
          key: type,
          value: count,
        }))
    : [];

  // Experiential drift data
  const driftData = drift?.drift_by_trait
    ? Object.entries(drift.drift_by_trait).map(([trait, info]) => ({
        trait: trait.replace(/_/g, ' '),
        mean_drift: Math.round(info.mean_drift * 1000) / 1000,
        agents_affected: info.agents_affected,
      }))
    : [];

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-gray-100">Inner Life & Experiential Mind</h1>

      {/* Summary Stats */}
      {pqDist?.stats && (
        <div className="grid grid-cols-2 gap-4 lg:grid-cols-5">
          <StatCard label="Mean PQ" value={pqDist.stats.mean.toFixed(3)} />
          <StatCard label="Std Dev" value={pqDist.stats.std.toFixed(3)} />
          <StatCard label="Min PQ" value={pqDist.stats.min.toFixed(3)} />
          <StatCard label="Max PQ" value={pqDist.stats.max.toFixed(3)} />
          <StatCard label="Agents" value={pqDist.stats.count} />
        </div>
      )}

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* PQ Distribution */}
        {pqData.length > 0 && (
          <div className="rounded-lg border border-gray-800 bg-gray-900 p-4 lg:col-span-2">
            <h2 className="mb-3 text-lg font-semibold text-gray-200">Phenomenal Quality Distribution</h2>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={pqData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="bucket" stroke="#9ca3af" tick={{ fontSize: 12 }} />
                <YAxis stroke="#9ca3af" tick={{ fontSize: 11 }} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }}
                  formatter={((value: any) => [value, 'Agents']) as any}
                />
                <Bar dataKey="count" name="Agents">
                  {pqData.map((entry) => (
                    <Cell key={entry.bucket} fill={PQ_BUCKET_COLORS[entry.bucket] ?? '#6B7280'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Population Mood Radar */}
        {radarData.length > 0 && (
          <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
            <h2 className="mb-3 text-lg font-semibold text-gray-200">Population Mood</h2>
            <ResponsiveContainer width="100%" height={280}>
              <RadarChart data={radarData}>
                <PolarGrid stroke="#374151" />
                <PolarAngleAxis dataKey="dimension" stroke="#9ca3af" tick={{ fontSize: 11 }} />
                <PolarRadiusAxis stroke="#4b5563" tick={{ fontSize: 10 }} domain={[0, 1]} />
                <Radar
                  name="Mood"
                  dataKey="value"
                  stroke="#3B82F6"
                  fill="#3B82F6"
                  fillOpacity={0.3}
                />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Event Type Breakdown */}
        {eventData.length > 0 && (
          <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
            <h2 className="mb-3 text-lg font-semibold text-gray-200">Event Types</h2>
            <ResponsiveContainer width="100%" height={280}>
              <PieChart>
                <Pie
                  data={eventData}
                  cx="50%"
                  cy="50%"
                  outerRadius={100}
                  dataKey="value"
                  nameKey="name"
                  label={((props: any) => `${props.name ?? ''} ${((props.percent ?? 0) * 100).toFixed(0)}%`) as any}
                  labelLine={false}
                >
                  {eventData.map((entry) => (
                    <Cell key={entry.key} fill={EVENT_COLORS[entry.key] ?? '#6B7280'} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }}
                  formatter={((value: any) => [value, 'Events']) as any}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Experiential Drift by Trait */}
        {driftData.length > 0 && (
          <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
            <h2 className="mb-3 text-lg font-semibold text-gray-200">Experiential Drift by Trait</h2>
            <ResponsiveContainer width="100%" height={Math.max(250, driftData.length * 28)}>
              <BarChart data={driftData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis type="number" stroke="#9ca3af" tick={{ fontSize: 11 }} />
                <YAxis type="category" dataKey="trait" stroke="#9ca3af" tick={{ fontSize: 10 }} width={100} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }}
                  formatter={((value: any, _name: string, item: any) => [
                    `${Number(value).toFixed(4)} (${item?.payload?.agents_affected ?? 0} agents)`,
                    'Mean Drift',
                  ]) as any}
                />
                <Bar dataKey="mean_drift" name="Mean Drift">
                  {driftData.map((entry, i) => (
                    <Cell key={i} fill={entry.mean_drift >= 0 ? '#22C55E' : '#EF4444'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Mood Map Scatter */}
        {moodMap?.agents && moodMap.agents.length > 0 && (
          <div className="rounded-lg border border-gray-800 bg-gray-900 p-4 lg:col-span-2">
            <h2 className="mb-3 text-lg font-semibold text-gray-200">Mood Map (Valence vs Arousal)</h2>
            <MoodScatter agents={moodMap.agents} />
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

function MoodScatter({ agents }: { agents: { agent_id: string; mood: Record<string, number>; phenomenal_quality: number }[] }) {
  const svgRef = useRef<SVGSVGElement>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!svgRef.current || agents.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const width = svgRef.current.clientWidth;
    const height = 220;
    const margin = { top: 10, right: 20, bottom: 35, left: 45 };
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;

    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    const xScale = d3.scaleLinear().domain([-1, 1]).range([0, innerW]);
    const yScale = d3.scaleLinear().domain([0, 1]).range([innerH, 0]);

    // Axes
    g.append('g')
      .attr('transform', `translate(0,${innerH})`)
      .call(d3.axisBottom(xScale).ticks(5))
      .selectAll('text').attr('fill', '#9ca3af').style('font-size', '10px');
    g.append('g')
      .call(d3.axisLeft(yScale).ticks(5))
      .selectAll('text').attr('fill', '#9ca3af').style('font-size', '10px');

    // Axis labels
    g.append('text').attr('x', innerW / 2).attr('y', innerH + 30).attr('fill', '#9ca3af')
      .attr('text-anchor', 'middle').style('font-size', '11px').text('Valence');
    g.append('text').attr('x', -innerH / 2).attr('y', -35).attr('fill', '#9ca3af')
      .attr('text-anchor', 'middle').attr('transform', 'rotate(-90)').style('font-size', '11px').text('Arousal');

    // Grid lines
    g.selectAll('line.grid-x').data(xScale.ticks(5)).enter()
      .append('line').attr('x1', (d: number) => xScale(d)).attr('x2', (d: number) => xScale(d))
      .attr('y1', 0).attr('y2', innerH).attr('stroke', '#374151').attr('stroke-dasharray', '3 3');
    g.selectAll('line.grid-y').data(yScale.ticks(5)).enter()
      .append('line').attr('x1', 0).attr('x2', innerW)
      .attr('y1', (d: number) => yScale(d)).attr('y2', (d: number) => yScale(d))
      .attr('stroke', '#374151').attr('stroke-dasharray', '3 3');

    const pqColor = (pq: number) => {
      if (pq < 0.2) return '#EF4444';
      if (pq < 0.4) return '#F97316';
      if (pq < 0.6) return '#F59E0B';
      if (pq < 0.8) return '#22C55E';
      return '#10B981';
    };

    const tooltip = d3.select(tooltipRef.current);

    // Dots
    g.selectAll('circle')
      .data(agents)
      .enter()
      .append('circle')
      .attr('cx', (d) => xScale(d.mood.valence ?? 0))
      .attr('cy', (d) => yScale(d.mood.arousal ?? 0))
      .attr('r', (d) => 3 + d.phenomenal_quality * 5)
      .attr('fill', (d) => pqColor(d.phenomenal_quality))
      .attr('opacity', 0.7)
      .attr('stroke', '#1f2937')
      .attr('stroke-width', 0.5)
      .on('mouseover', (event, d) => {
        const dims = Object.entries(d.mood).map(([k, v]) => `${EXPERIENCE_DIM_LABELS[k] ?? k}: ${v.toFixed(2)}`).join(', ');
        tooltip
          .style('opacity', '1')
          .style('left', `${event.offsetX + 12}px`)
          .style('top', `${event.offsetY - 10}px`)
          .html(`<div class="text-xs"><strong>Agent ${d.agent_id}</strong><br/>PQ: ${d.phenomenal_quality.toFixed(3)}<br/>${dims}</div>`);
      })
      .on('mouseout', () => {
        tooltip.style('opacity', '0');
      });

  }, [agents]);

  return (
    <div className="relative">
      <svg ref={svgRef} width="100%" height={220} />
      <div
        ref={tooltipRef}
        className="pointer-events-none absolute rounded border border-gray-700 bg-gray-800 px-2 py-1 opacity-0 transition-opacity"
        style={{ opacity: 0 }}
      />
    </div>
  );
}

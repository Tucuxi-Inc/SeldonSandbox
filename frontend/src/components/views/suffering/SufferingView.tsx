import { useState, useEffect, useRef } from 'react';
import { useSimulationStore } from '../../../stores/simulation';
import { EmptyState } from '../../shared/EmptyState';
import { REGION_COLORS, REGION_LABELS, REGION_ORDER } from '../../../lib/constants';
import * as api from '../../../api/client';
import type { AgentDetail } from '../../../types';
import * as d3 from 'd3';

export function SufferingView() {
  const { activeSessionId, agents } = useSimulationStore();
  const svgRef = useRef<SVGSVGElement>(null);
  const [selectedAgent, setSelectedAgent] = useState<AgentDetail | null>(null);

  useEffect(() => {
    if (!svgRef.current || agents.length === 0) return;

    const svg = d3.select(svgRef.current);
    const width = svgRef.current.clientWidth;
    const height = 500;
    const margin = { top: 20, right: 30, bottom: 50, left: 60 };

    svg.selectAll('*').remove();
    svg.attr('width', width).attr('height', height);

    const xScale = d3
      .scaleLinear()
      .domain([0, d3.max(agents, (a) => a.suffering) || 1])
      .range([margin.left, width - margin.right])
      .nice();

    const yScale = d3
      .scaleLinear()
      .domain([0, d3.max(agents, (a) => a.latest_contribution) || 1])
      .range([height - margin.bottom, margin.top])
      .nice();

    // Axes
    svg.append('g')
      .attr('transform', `translate(0, ${height - margin.bottom})`)
      .call(d3.axisBottom(xScale).ticks(8))
      .call((g) => g.selectAll('text').attr('fill', '#9ca3af').style('font-size', '11px'))
      .call((g) => g.selectAll('line').attr('stroke', '#374151'))
      .call((g) => g.select('.domain').attr('stroke', '#374151'));

    svg.append('g')
      .attr('transform', `translate(${margin.left}, 0)`)
      .call(d3.axisLeft(yScale).ticks(8))
      .call((g) => g.selectAll('text').attr('fill', '#9ca3af').style('font-size', '11px'))
      .call((g) => g.selectAll('line').attr('stroke', '#374151'))
      .call((g) => g.select('.domain').attr('stroke', '#374151'));

    // Axis labels
    svg.append('text')
      .attr('x', width / 2).attr('y', height - 8)
      .attr('fill', '#6b7280').attr('text-anchor', 'middle').style('font-size', '12px')
      .text('Suffering');

    svg.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -height / 2).attr('y', 16)
      .attr('fill', '#6b7280').attr('text-anchor', 'middle').style('font-size', '12px')
      .text('Contribution');

    // Grid
    svg.append('g')
      .attr('transform', `translate(${margin.left}, 0)`)
      .call(d3.axisLeft(yScale).ticks(8).tickSize(-(width - margin.left - margin.right)).tickFormat(() => ''))
      .call((g) => g.selectAll('line').attr('stroke', '#1f2937'))
      .call((g) => g.select('.domain').remove());

    // Dots
    svg.selectAll('circle')
      .data(agents)
      .join('circle')
      .attr('cx', (d) => xScale(d.suffering))
      .attr('cy', (d) => yScale(d.latest_contribution))
      .attr('r', 4)
      .attr('fill', (d) => REGION_COLORS[d.processing_region] || '#6B7280')
      .attr('opacity', 0.7)
      .attr('cursor', 'pointer')
      .on('click', async (_event, d) => {
        if (!activeSessionId) return;
        const detail = await api.getAgentDetail(activeSessionId, d.id);
        setSelectedAgent(detail);
      })
      .append('title')
      .text((d) => `${d.name} (${d.processing_region})\nSuffering: ${d.suffering.toFixed(3)}\nContribution: ${d.latest_contribution.toFixed(3)}`);

    // Trajectory overlay
    if (selectedAgent && selectedAgent.suffering_history.length > 0) {
      const trajectoryData = selectedAgent.suffering_history.map((s, i) => ({
        suffering: s,
        contribution: selectedAgent.contribution_history[i] || 0,
      }));

      const lineGen = d3.line<{ suffering: number; contribution: number }>()
        .x((d) => xScale(d.suffering))
        .y((d) => yScale(d.contribution));

      svg.append('path')
        .datum(trajectoryData)
        .attr('fill', 'none')
        .attr('stroke', '#ffffff')
        .attr('stroke-width', 2)
        .attr('stroke-dasharray', '4,4')
        .attr('d', lineGen);

      // Start and end markers
      if (trajectoryData.length > 0) {
        const start = trajectoryData[0];
        const end = trajectoryData[trajectoryData.length - 1];
        svg.append('circle')
          .attr('cx', xScale(start.suffering)).attr('cy', yScale(start.contribution))
          .attr('r', 6).attr('fill', '#10B981').attr('stroke', '#fff').attr('stroke-width', 2);
        svg.append('circle')
          .attr('cx', xScale(end.suffering)).attr('cy', yScale(end.contribution))
          .attr('r', 6).attr('fill', '#EF4444').attr('stroke', '#fff').attr('stroke-width', 2);
      }
    }
  }, [agents, selectedAgent, activeSessionId]);

  if (agents.length === 0) {
    return <EmptyState message="Run a simulation to see suffering vs. contribution data" />;
  }

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-gray-100">Suffering vs. Contribution</h1>

      {/* Legend */}
      <div className="flex flex-wrap gap-4">
        {REGION_ORDER.map((region) => (
          <div key={region} className="flex items-center gap-2">
            <div className="h-3 w-3 rounded-full" style={{ backgroundColor: REGION_COLORS[region] }} />
            <span className="text-sm text-gray-400">{REGION_LABELS[region]}</span>
          </div>
        ))}
      </div>

      {/* Chart */}
      <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
        <svg ref={svgRef} className="w-full" style={{ minHeight: 500 }} />
      </div>

      {/* Selected Agent Info */}
      {selectedAgent && (
        <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold text-gray-200">
              {selectedAgent.name}
              <span className="ml-2 text-sm text-gray-500">({selectedAgent.processing_region})</span>
            </h3>
            <button
              onClick={() => setSelectedAgent(null)}
              className="text-sm text-gray-500 hover:text-gray-300"
            >
              Clear
            </button>
          </div>
          <div className="mt-2 grid grid-cols-4 gap-4 text-sm">
            <div><span className="text-gray-500">Age:</span> <span className="text-gray-200">{selectedAgent.age}</span></div>
            <div><span className="text-gray-500">Gen:</span> <span className="text-gray-200">{selectedAgent.generation}</span></div>
            <div><span className="text-gray-500">Birth Order:</span> <span className="text-gray-200">{selectedAgent.birth_order}</span></div>
            <div><span className="text-gray-500">Outsider:</span> <span className="text-gray-200">{selectedAgent.is_outsider ? 'Yes' : 'No'}</span></div>
          </div>
          <p className="mt-1 text-xs text-gray-600">
            Green dot = start of life, Red dot = current position. Dashed line = life trajectory.
          </p>
        </div>
      )}
    </div>
  );
}

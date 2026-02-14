import { useState, useEffect, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import * as d3 from 'd3';
import { useSimulationStore } from '../../../stores/simulation';
import * as api from '../../../api/client';
import type { HexGridResponse, HexTileData, TickStateResponse, AgentTickActivity } from '../../../types';
import { TERRAIN_COLORS, REGION_COLORS, LIFE_PHASE_SIZES, SEASON_COLORS } from '../../../lib/constants';

const HEX_SIZE = 28;
const HEX_WIDTH = HEX_SIZE * 2;
const HEX_HEIGHT = Math.sqrt(3) * HEX_SIZE;

function hexCenter(q: number, r: number): [number, number] {
  const x = q * HEX_WIDTH * 0.75;
  const y = (r + q * 0.5) * HEX_HEIGHT;
  return [x, y];
}

function hexPoints(): string {
  const pts: string[] = [];
  for (let i = 0; i < 6; i++) {
    const angle = (Math.PI / 180) * (60 * i);
    pts.push(`${HEX_SIZE * Math.cos(angle)},${HEX_SIZE * Math.sin(angle)}`);
  }
  return pts.join(' ');
}

const HEX_POLYGON_POINTS = hexPoints();

const SEASON_LABELS: Record<string, string> = {
  spring: 'Spring', summer: 'Summer', autumn: 'Autumn', winter: 'Winter',
};

export function WorldPreview() {
  const { activeSessionId, sessions } = useSimulationStore();
  const navigate = useNavigate();
  const active = sessions.find(s => s.id === activeSessionId);

  const [gridData, setGridData] = useState<HexGridResponse | null>(null);
  const [tickState, setTickState] = useState<TickStateResponse | null>(null);
  const [loading, setLoading] = useState(false);

  const svgRef = useRef<SVGSVGElement>(null);
  const gRef = useRef<d3.Selection<SVGGElement, unknown, null, undefined> | null>(null);
  const initializedRef = useRef(false);
  const prevSessionRef = useRef<string | null>(null);

  // Fetch grid data and tick state when session changes or generation advances
  useEffect(() => {
    if (!activeSessionId) {
      setGridData(null);
      setTickState(null);
      initializedRef.current = false;
      prevSessionRef.current = null;
      return;
    }

    // Reset D3 on session change
    if (prevSessionRef.current !== activeSessionId) {
      initializedRef.current = false;
      prevSessionRef.current = activeSessionId;
    }

    setLoading(true);
    Promise.all([
      api.getHexGrid(activeSessionId),
      api.getTickState(activeSessionId),
    ]).then(([grid, tick]) => {
      setGridData(grid.enabled ? grid : null);
      setTickState(tick.enabled ? tick : null);
      setLoading(false);
    }).catch(() => {
      setLoading(false);
    });
  }, [activeSessionId, active?.current_generation, active?.population_size]);

  // Initialize terrain when gridData changes
  const initializeTerrain = useCallback(() => {
    if (!svgRef.current || !gridData?.tiles || initializedRef.current) return;

    const svg = d3.select(svgRef.current);
    const container = svgRef.current.parentElement;
    const width = container?.clientWidth ?? 600;
    const height = container?.clientHeight ?? 300;

    svg.attr('width', width).attr('height', height);
    svg.selectAll('*').remove();

    // Season overlay
    svg.append('rect')
      .attr('class', 'season-overlay')
      .attr('width', width)
      .attr('height', height)
      .attr('fill', 'transparent')
      .style('pointer-events', 'none');

    const g = svg.append('g').attr('class', 'preview-root');
    gRef.current = g;

    g.append('g').attr('class', 'terrain-layer');
    g.append('g').attr('class', 'agent-layer');

    // Fit grid to container (no interactive zoom — this is a preview)
    const allCenters = gridData.tiles.map(t => hexCenter(t.q, t.r));
    const minX = Math.min(...allCenters.map(c => c[0]));
    const maxX = Math.max(...allCenters.map(c => c[0]));
    const minY = Math.min(...allCenters.map(c => c[1]));
    const maxY = Math.max(...allCenters.map(c => c[1]));
    const gridW = maxX - minX + HEX_WIDTH;
    const gridH = maxY - minY + HEX_HEIGHT;
    const scale = Math.min(width / gridW, height / gridH) * 0.9;
    const tx = (width - gridW * scale) / 2 - minX * scale + HEX_SIZE * scale;
    const ty = (height - gridH * scale) / 2 - minY * scale + HEX_SIZE * scale;
    g.attr('transform', `translate(${tx}, ${ty}) scale(${scale})`);

    // Draw terrain hexes
    g.select('.terrain-layer')
      .selectAll('.hex-tile')
      .data(gridData.tiles, (d: any) => `${d.q},${d.r}`)
      .join('g')
      .attr('class', 'hex-tile')
      .attr('transform', (d: HexTileData) => {
        const [x, y] = hexCenter(d.q, d.r);
        return `translate(${x}, ${y})`;
      })
      .each(function (d: HexTileData) {
        d3.select(this).append('polygon')
          .attr('points', HEX_POLYGON_POINTS)
          .attr('fill', TERRAIN_COLORS[d.terrain_type] || '#333')
          .attr('stroke', '#1a1a2e')
          .attr('stroke-width', 0.5);
      });

    initializedRef.current = true;
  }, [gridData]);

  useEffect(() => {
    initializeTerrain();
  }, [initializeTerrain]);

  // Update season overlay
  useEffect(() => {
    if (!svgRef.current || !tickState) return;
    d3.select(svgRef.current).select('.season-overlay')
      .attr('fill', SEASON_COLORS[tickState.season] || 'transparent');
  }, [tickState?.season]);

  // Render agents
  useEffect(() => {
    if (!gRef.current || !tickState?.agent_activities) return;

    const agentLayer = gRef.current.select('.agent-layer');
    const activities = tickState.agent_activities;

    const agentData = Object.values(activities).filter(
      (a: AgentTickActivity) => a.location != null,
    ) as AgentTickActivity[];

    const circles = agentLayer
      .selectAll<SVGCircleElement, AgentTickActivity>('.agent-dot')
      .data(agentData, (d: AgentTickActivity) => d.agent_id);

    circles.enter()
      .append('circle')
      .attr('class', 'agent-dot')
      .attr('cx', d => hexCenter(d.location![0], d.location![1])[0])
      .attr('cy', d => hexCenter(d.location![0], d.location![1])[1])
      .attr('r', d => (LIFE_PHASE_SIZES[d.life_phase] || 4) * 0.8)
      .attr('fill', d => REGION_COLORS[d.processing_region] || '#6B7280')
      .attr('stroke', '#000')
      .attr('stroke-width', 0.3)
      .attr('opacity', 0.85);

    circles
      .attr('cx', d => hexCenter(d.location![0], d.location![1])[0])
      .attr('cy', d => hexCenter(d.location![0], d.location![1])[1])
      .attr('r', d => (LIFE_PHASE_SIZES[d.life_phase] || 4) * 0.8)
      .attr('fill', d => REGION_COLORS[d.processing_region] || '#6B7280');

    circles.exit().remove();
  }, [tickState]);

  // Don't render if no tick-enabled session
  if (!activeSessionId || !gridData || loading) return null;

  return (
    <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="text-lg font-semibold text-gray-200">World Preview</h2>
        <div className="flex items-center gap-3">
          {tickState && (
            <div className="flex items-center gap-2 text-xs text-gray-400">
              <span>Year {tickState.year}</span>
              <span
                className="rounded px-1.5 py-0.5 text-xs font-medium"
                style={{
                  backgroundColor: SEASON_COLORS[tickState.season]?.replace('0.05', '0.3')
                    || 'transparent',
                  color: '#e5e7eb',
                }}
              >
                {SEASON_LABELS[tickState.season] || tickState.season}
              </span>
              <span>Pop: {tickState.population_count}</span>
            </div>
          )}
          <button
            onClick={() => navigate('/world')}
            className="rounded-md bg-blue-600/20 px-3 py-1 text-xs font-medium text-blue-400 hover:bg-blue-600/30 transition-colors"
          >
            Open World View
          </button>
        </div>
      </div>
      <div className="relative overflow-hidden rounded-md border border-gray-800 bg-gray-950" style={{ height: 280 }}>
        <svg ref={svgRef} className="w-full h-full" />
      </div>
    </div>
  );
}

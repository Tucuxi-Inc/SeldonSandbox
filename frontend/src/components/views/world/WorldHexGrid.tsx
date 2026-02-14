import { useEffect, useRef, useCallback } from 'react';
import * as d3 from 'd3';
import type { HexGridResponse, HexTileData, TickStateResponse, AgentTickActivity } from '../../../types';
import {
  TERRAIN_COLORS,
  REGION_COLORS,
  ACTIVITY_ICONS,
  LIFE_PHASE_SIZES,
  SEASON_COLORS,
} from '../../../lib/constants';

type ColorMode = 'terrain' | 'density' | 'region';

interface WorldHexGridProps {
  gridData: HexGridResponse;
  tickState: TickStateResponse | null;
  colorMode: ColorMode;
  selectedAgentId: string | null;
  showConnections: boolean;
  tickHistory: Map<string, { locations: (number[] | null)[] }>;
  onSelectAgent: (id: string | null) => void;
}

const HEX_SIZE = 28;
const HEX_WIDTH = HEX_SIZE * 2;
const HEX_HEIGHT = Math.sqrt(3) * HEX_SIZE;

function hexCenter(q: number, r: number): [number, number] {
  const x = q * HEX_WIDTH * 0.75;
  const y = (r + q * 0.5) * HEX_HEIGHT;
  return [x, y];
}

function hexPoints(): string {
  const points: string[] = [];
  for (let i = 0; i < 6; i++) {
    const angle = (Math.PI / 180) * (60 * i);
    points.push(`${HEX_SIZE * Math.cos(angle)},${HEX_SIZE * Math.sin(angle)}`);
  }
  return points.join(' ');
}

const HEX_POLYGON_POINTS = hexPoints();

export function WorldHexGrid({
  gridData,
  tickState,
  colorMode,
  selectedAgentId,
  showConnections,
  tickHistory,
  onSelectAgent,
}: WorldHexGridProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const gRef = useRef<d3.Selection<SVGGElement, unknown, null, undefined> | null>(null);
  const zoomRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null);
  const initializedRef = useRef(false);
  const tooltipRef = useRef<HTMLDivElement>(null);

  // Build a location lookup for tiles
  const tileLookup = useRef(new Map<string, HexTileData>());
  useEffect(() => {
    const map = new Map<string, HexTileData>();
    for (const tile of gridData.tiles || []) {
      map.set(`${tile.q},${tile.r}`, tile);
    }
    tileLookup.current = map;
  }, [gridData]);

  // Initialize SVG, terrain layer, and zoom once
  const initializeGrid = useCallback(() => {
    if (!svgRef.current || !gridData.tiles || initializedRef.current) return;

    const svg = d3.select(svgRef.current);
    const container = svgRef.current.parentElement;
    const width = container?.clientWidth ?? 800;
    const height = container?.clientHeight ?? 600;

    svg.attr('width', width).attr('height', height);
    svg.selectAll('*').remove();

    // Season overlay rect (behind everything)
    svg.append('rect')
      .attr('class', 'season-overlay')
      .attr('width', width)
      .attr('height', height)
      .attr('fill', 'transparent')
      .style('pointer-events', 'none');

    const g = svg.append('g').attr('class', 'world-root');
    gRef.current = g;

    // Terrain layer
    const terrainLayer = g.append('g').attr('class', 'terrain-layer');
    // Connection lines layer (between terrain and agents)
    g.append('g').attr('class', 'connection-layer');
    // Trail layer (selected agent history)
    g.append('g').attr('class', 'trail-layer');
    // Agent layer
    g.append('g').attr('class', 'agent-layer');
    // Activity label layer
    g.append('g').attr('class', 'activity-layer');

    // Zoom
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.3, 6])
      .on('zoom', (event) => {
        g.attr('transform', event.transform.toString());
      });
    zoomRef.current = zoom;
    svg.call(zoom);

    // Center the grid
    const allCenters = gridData.tiles.map(t => hexCenter(t.q, t.r));
    const minX = Math.min(...allCenters.map(c => c[0]));
    const maxX = Math.max(...allCenters.map(c => c[0]));
    const minY = Math.min(...allCenters.map(c => c[1]));
    const maxY = Math.max(...allCenters.map(c => c[1]));
    const gridW = maxX - minX + HEX_WIDTH;
    const gridH = maxY - minY + HEX_HEIGHT;
    const scale = Math.min(width / gridW, height / gridH) * 0.85;
    const tx = (width - gridW * scale) / 2 - minX * scale + HEX_SIZE * scale;
    const ty = (height - gridH * scale) / 2 - minY * scale + HEX_SIZE * scale;
    svg.call(zoom.transform, d3.zoomIdentity.translate(tx, ty).scale(scale));

    // Draw terrain hexes
    terrainLayer.selectAll('.hex-tile')
      .data(gridData.tiles, (d: any) => `${d.q},${d.r}`)
      .join('g')
      .attr('class', 'hex-tile')
      .attr('transform', d => {
        const [x, y] = hexCenter(d.q, d.r);
        return `translate(${x}, ${y})`;
      })
      .each(function (d) {
        const el = d3.select(this);
        el.append('polygon')
          .attr('points', HEX_POLYGON_POINTS)
          .attr('fill', TERRAIN_COLORS[d.terrain_type] || '#333')
          .attr('stroke', '#1a1a2e')
          .attr('stroke-width', 0.5)
          .attr('cursor', 'pointer')
          .on('click', () => onSelectAgent(null))
          .on('mouseenter', (event: MouseEvent) => {
            const tooltip = tooltipRef.current;
            if (!tooltip) return;
            tooltip.style.display = 'block';
            tooltip.style.left = `${event.pageX + 12}px`;
            tooltip.style.top = `${event.pageY - 10}px`;
            tooltip.innerHTML = `
              <div class="font-medium">${d.terrain_type}</div>
              <div class="text-xs text-gray-400">(${d.q}, ${d.r})</div>
              <div class="text-xs mt-1">Habitability: ${(d.habitability * 100).toFixed(0)}%</div>
            `;
          })
          .on('mouseleave', () => {
            const tooltip = tooltipRef.current;
            if (tooltip) tooltip.style.display = 'none';
          });
      });

    initializedRef.current = true;
  }, [gridData, onSelectAgent]);

  useEffect(() => {
    initializeGrid();
  }, [initializeGrid]);

  // Update terrain colors when colorMode changes
  useEffect(() => {
    if (!gRef.current || !gridData.tiles) return;

    const maxAgents = Math.max(1, ...gridData.tiles.map(t => t.agent_count));
    const densityScale = d3.scaleSequential(d3.interpolateYlOrRd).domain([0, maxAgents]);

    gRef.current.select('.terrain-layer')
      .selectAll<SVGGElement, HexTileData>('.hex-tile')
      .select('polygon')
      .attr('fill', (d: HexTileData) => {
        if (colorMode === 'density') {
          return d.agent_count > 0 ? densityScale(d.agent_count) : '#1a1a2e';
        }
        if (colorMode === 'region') {
          // Find dominant region for this tile
          const counts = d.region_counts || {};
          const top = Object.entries(counts).sort((a, b) => b[1] - a[1])[0];
          return top ? (REGION_COLORS[top[0]] || '#333') + 'aa' : '#1a1a2e';
        }
        return TERRAIN_COLORS[d.terrain_type] || '#333';
      });
  }, [colorMode, gridData]);

  // Update season overlay
  useEffect(() => {
    if (!svgRef.current || !tickState) return;
    d3.select(svgRef.current).select('.season-overlay')
      .transition()
      .duration(800)
      .attr('fill', SEASON_COLORS[tickState.season] || 'transparent');
  }, [tickState?.season]);

  // Animate agents when tickState changes
  useEffect(() => {
    if (!gRef.current || !tickState?.agent_activities) return;

    const agentLayer = gRef.current.select('.agent-layer');
    const activityLayer = gRef.current.select('.activity-layer');
    const activities = tickState.agent_activities;
    const agentNames = tickState.agent_names || {};

    // Build data array from activities (only agents with locations)
    const agentData = Object.values(activities).filter(
      (a: AgentTickActivity) => a.location != null,
    ) as AgentTickActivity[];

    // Agent circles with D3 data join + transitions
    const circles = agentLayer
      .selectAll<SVGCircleElement, AgentTickActivity>('.agent-dot')
      .data(agentData, (d: AgentTickActivity) => d.agent_id);

    // Enter: new agents (births) fade in
    const enter = circles.enter()
      .append('circle')
      .attr('class', 'agent-dot')
      .attr('cx', d => {
        const loc = d.location!;
        return hexCenter(loc[0], loc[1])[0];
      })
      .attr('cy', d => {
        const loc = d.location!;
        return hexCenter(loc[0], loc[1])[1];
      })
      .attr('r', 0)
      .attr('fill', d => REGION_COLORS[d.processing_region] || '#6B7280')
      .attr('stroke', d => d.agent_id === selectedAgentId ? '#fff' : '#000')
      .attr('stroke-width', d => d.agent_id === selectedAgentId ? 1.5 : 0.5)
      .attr('cursor', 'pointer')
      .attr('opacity', 0)
      .on('click', (_event: MouseEvent, d: AgentTickActivity) => {
        onSelectAgent(d.agent_id === selectedAgentId ? null : d.agent_id);
      });

    enter.transition()
      .duration(400)
      .attr('r', d => LIFE_PHASE_SIZES[d.life_phase] || 4)
      .attr('opacity', 0.9);

    // Update: existing agents transition to new position
    circles
      .transition()
      .duration(600)
      .ease(d3.easeQuadInOut)
      .attr('cx', d => {
        const loc = d.location!;
        return hexCenter(loc[0], loc[1])[0];
      })
      .attr('cy', d => {
        const loc = d.location!;
        return hexCenter(loc[0], loc[1])[1];
      })
      .attr('r', d => LIFE_PHASE_SIZES[d.life_phase] || 4)
      .attr('fill', d => REGION_COLORS[d.processing_region] || '#6B7280')
      .attr('stroke', d => d.agent_id === selectedAgentId ? '#fff' : '#000')
      .attr('stroke-width', d => d.agent_id === selectedAgentId ? 1.5 : 0.5);

    // Exit: dead agents shrink out
    circles.exit()
      .transition()
      .duration(400)
      .attr('r', 0)
      .attr('opacity', 0)
      .remove();

    // Activity icons (visible as small text above agents)
    const iconData = agentData.filter(a => a.activity != null);

    const icons = activityLayer
      .selectAll<SVGTextElement, AgentTickActivity>('.activity-icon')
      .data(iconData, (d: AgentTickActivity) => d.agent_id);

    icons.enter()
      .append('text')
      .attr('class', 'activity-icon')
      .attr('text-anchor', 'middle')
      .attr('font-size', '8px')
      .attr('pointer-events', 'none')
      .attr('opacity', 0)
      .merge(icons as any)
      .transition()
      .duration(600)
      .ease(d3.easeQuadInOut)
      .attr('x', d => {
        const loc = d.location!;
        return hexCenter(loc[0], loc[1])[0];
      })
      .attr('y', d => {
        const loc = d.location!;
        return hexCenter(loc[0], loc[1])[1] - (LIFE_PHASE_SIZES[d.life_phase] || 4) - 3;
      })
      .attr('opacity', 0.8)
      .text(d => ACTIVITY_ICONS[d.activity!] || '');

    icons.exit().remove();

    // Agent name labels (small text below agents)
    const nameData = agentData;
    const names = activityLayer
      .selectAll<SVGTextElement, AgentTickActivity>('.agent-name')
      .data(nameData, (d: AgentTickActivity) => `name-${d.agent_id}`);

    names.enter()
      .append('text')
      .attr('class', 'agent-name')
      .attr('text-anchor', 'middle')
      .attr('font-size', '5px')
      .attr('fill', '#9CA3AF')
      .attr('pointer-events', 'none')
      .attr('opacity', 0)
      .merge(names as any)
      .transition()
      .duration(600)
      .ease(d3.easeQuadInOut)
      .attr('x', d => {
        const loc = d.location!;
        return hexCenter(loc[0], loc[1])[0];
      })
      .attr('y', d => {
        const loc = d.location!;
        return hexCenter(loc[0], loc[1])[1] + (LIFE_PHASE_SIZES[d.life_phase] || 4) + 6;
      })
      .attr('opacity', 0.6)
      .text(d => agentNames[d.agent_id] || '');

    names.exit().remove();

    // --- Connection lines between partners ---
    const connectionLayer = gRef.current.select('.connection-layer');
    connectionLayer.selectAll('*').remove();

    if (showConnections) {
      // Build partner pairs from agents list (use gridData.tiles agent info + living agents)
      // We use the current grid agent list to find agents on tiles
      const agentLocMap = new Map<string, number[]>();
      for (const ata of agentData) {
        if (ata.location) agentLocMap.set(ata.agent_id, ata.location);
      }

      // Draw lines for agents that share a tile and are in the same tile's agent list
      // For a proper partner detection we'd need partner_id from the backend,
      // but we can draw proximity lines for agents on same or adjacent tiles
      // For now, draw lines between agents on the same hex (social clusters)
      const tileGroups = new Map<string, AgentTickActivity[]>();
      for (const ata of agentData) {
        if (!ata.location) continue;
        const key = `${ata.location[0]},${ata.location[1]}`;
        const group = tileGroups.get(key) || [];
        group.push(ata);
        tileGroups.set(key, group);
      }

      for (const [, group] of tileGroups) {
        if (group.length < 2) continue;
        // Draw faint lines between first few agents on same tile
        for (let i = 0; i < Math.min(group.length - 1, 3); i++) {
          const a = group[i];
          const b = group[i + 1];
          if (!a.location || !b.location) continue;
          const [ax, ay] = hexCenter(a.location[0], a.location[1]);
          const [bx, by] = hexCenter(b.location[0], b.location[1]);
          // Offset slightly so lines don't all overlap
          const offsetA = (i - 1) * 2;
          connectionLayer.append('line')
            .attr('x1', ax + offsetA)
            .attr('y1', ay + offsetA)
            .attr('x2', bx + offsetA)
            .attr('y2', by + offsetA)
            .attr('stroke', '#EF4444')
            .attr('stroke-width', 0.5)
            .attr('stroke-dasharray', '2,2')
            .attr('opacity', 0.3)
            .attr('pointer-events', 'none');
        }
      }
    }

    // --- Selected agent trail ---
    const trailLayer = gRef.current.select('.trail-layer');
    trailLayer.selectAll('*').remove();

    if (selectedAgentId) {
      const history = tickHistory.get(selectedAgentId);
      if (history && history.locations.length > 1) {
        const locs = history.locations.filter(l => l != null) as number[][];
        locs.forEach((loc, i) => {
          const opacity = 0.1 + (i / locs.length) * 0.4;
          const [cx, cy] = hexCenter(loc[0], loc[1]);
          trailLayer.append('circle')
            .attr('cx', cx)
            .attr('cy', cy)
            .attr('r', 2)
            .attr('fill', '#fff')
            .attr('opacity', opacity)
            .attr('pointer-events', 'none');
        });
        // Draw line connecting trail points
        if (locs.length >= 2) {
          const lineGen = d3.line<number[]>()
            .x(d => hexCenter(d[0], d[1])[0])
            .y(d => hexCenter(d[0], d[1])[1])
            .curve(d3.curveCatmullRom);
          trailLayer.append('path')
            .attr('d', lineGen(locs)!)
            .attr('fill', 'none')
            .attr('stroke', '#fff')
            .attr('stroke-width', 0.8)
            .attr('opacity', 0.2)
            .attr('pointer-events', 'none');
        }
      }
    }

  }, [tickState, selectedAgentId, showConnections, tickHistory, onSelectAgent]);

  return (
    <div className="relative flex-1 rounded-lg border border-gray-800 bg-gray-950 overflow-hidden">
      <svg ref={svgRef} className="w-full h-full" style={{ minHeight: 500 }} />
      {/* Tooltip */}
      <div
        ref={tooltipRef}
        className="pointer-events-none fixed z-50 rounded-lg border border-gray-700 bg-gray-900 px-3 py-2 text-sm text-gray-200 shadow-lg"
        style={{ display: 'none' }}
      />
    </div>
  );
}

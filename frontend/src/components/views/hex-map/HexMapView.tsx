import { useState, useEffect, useRef, useCallback } from 'react';
import { useSimulationStore } from '../../../stores/simulation';
import { EmptyState } from '../../shared/EmptyState';
import { TERRAIN_COLORS, TERRAIN_LABELS, REGION_COLORS } from '../../../lib/constants';
import * as api from '../../../api/client';
import type { HexGridResponse, HexTileData, AgentSummary } from '../../../types';
import * as d3 from 'd3';

type ColorMode = 'terrain' | 'density' | 'habitability';

export function HexMapView() {
  const { activeSessionId } = useSimulationStore();
  const [gridData, setGridData] = useState<HexGridResponse | null>(null);
  const [colorMode, setColorMode] = useState<ColorMode>('terrain');
  const [selectedTile, setSelectedTile] = useState<HexTileData | null>(null);
  const [tileAgents, setTileAgents] = useState<AgentSummary[]>([]);
  const [loading, setLoading] = useState(false);
  const svgRef = useRef<SVGSVGElement>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);

  const loadGrid = useCallback(async () => {
    if (!activeSessionId) return;
    setLoading(true);
    try {
      const data = await api.getHexGrid(activeSessionId);
      setGridData(data);
    } catch {
      setGridData(null);
    }
    setLoading(false);
  }, [activeSessionId]);

  useEffect(() => {
    loadGrid();
  }, [loadGrid]);

  // Load tile detail when selected
  useEffect(() => {
    if (!activeSessionId || !selectedTile) {
      setTileAgents([]);
      return;
    }
    api.getHexTileDetail(activeSessionId, selectedTile.q, selectedTile.r)
      .then((detail) => {
        if (detail.agents) setTileAgents(detail.agents);
      })
      .catch(() => setTileAgents([]));
  }, [activeSessionId, selectedTile]);

  // D3 rendering
  useEffect(() => {
    if (!gridData?.enabled || !gridData.tiles || !svgRef.current) return;

    const svg = d3.select(svgRef.current);
    const container = svgRef.current.parentElement;
    const width = container?.clientWidth ?? 800;
    const height = container?.clientHeight ?? 600;

    svg.selectAll('*').remove();
    svg.attr('width', width).attr('height', height);

    const hexSize = 28;
    const hexWidth = hexSize * 2;
    const hexHeight = Math.sqrt(3) * hexSize;

    // Flat-top hex: x = q * 3/2 * size, y = (q/2 + r) * sqrt(3) * size
    function hexCenter(q: number, r: number): [number, number] {
      const x = q * hexWidth * 0.75;
      const y = (r + q * 0.5) * hexHeight;
      return [x, y];
    }

    // Flat-top hex points
    function hexPoints(cx: number, cy: number): string {
      const points: string[] = [];
      for (let i = 0; i < 6; i++) {
        const angle = (Math.PI / 180) * (60 * i);
        points.push(`${cx + hexSize * Math.cos(angle)},${cy + hexSize * Math.sin(angle)}`);
      }
      return points.join(' ');
    }

    // Color scales
    const maxAgents = Math.max(1, ...gridData.tiles.map(t => t.agent_count));
    const densityScale = d3.scaleSequential(d3.interpolateYlOrRd).domain([0, maxAgents]);
    const habitabilityScale = d3.scaleSequential(d3.interpolateGnBu).domain([0, 1]);

    function tileColor(tile: HexTileData): string {
      switch (colorMode) {
        case 'density':
          return tile.agent_count > 0 ? densityScale(tile.agent_count) : '#1a1a2e';
        case 'habitability':
          return habitabilityScale(tile.habitability);
        default:
          return TERRAIN_COLORS[tile.terrain_type] || '#333';
      }
    }

    // Create zoomable group
    const g = svg.append('g');

    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.3, 5])
      .on('zoom', (event) => {
        g.attr('transform', event.transform.toString());
      });

    svg.call(zoom);

    // Center the grid
    const allCenters = gridData.tiles.map(t => hexCenter(t.q, t.r));
    const minX = Math.min(...allCenters.map(c => c[0]));
    const maxX = Math.max(...allCenters.map(c => c[0]));
    const minY = Math.min(...allCenters.map(c => c[1]));
    const maxY = Math.max(...allCenters.map(c => c[1]));
    const gridW = maxX - minX + hexWidth;
    const gridH = maxY - minY + hexHeight;
    const scale = Math.min(width / gridW, height / gridH) * 0.85;
    const tx = (width - gridW * scale) / 2 - minX * scale + hexSize * scale;
    const ty = (height - gridH * scale) / 2 - minY * scale + hexSize * scale;

    svg.call(zoom.transform, d3.zoomIdentity.translate(tx, ty).scale(scale));

    // Draw hexagons
    const hexes = g.selectAll('.hex')
      .data(gridData.tiles)
      .join('g')
      .attr('class', 'hex')
      .attr('transform', d => {
        const [x, y] = hexCenter(d.q, d.r);
        return `translate(${x}, ${y})`;
      });

    hexes.append('polygon')
      .attr('points', () => hexPoints(0, 0))
      .attr('fill', d => tileColor(d))
      .attr('stroke', d => d === selectedTile ? '#fff' : '#1a1a2e')
      .attr('stroke-width', d => d === selectedTile ? 2 : 0.5)
      .attr('cursor', 'pointer')
      .on('click', (_event, d) => {
        setSelectedTile(prev => (prev?.q === d.q && prev?.r === d.r) ? null : d);
      })
      .on('mouseenter', (event, d) => {
        const tooltip = tooltipRef.current;
        if (!tooltip) return;
        tooltip.style.display = 'block';
        tooltip.style.left = `${event.pageX + 12}px`;
        tooltip.style.top = `${event.pageY - 10}px`;
        tooltip.innerHTML = `
          <div class="font-medium">${TERRAIN_LABELS[d.terrain_type] || d.terrain_type}</div>
          <div class="text-xs text-gray-400">(${d.q}, ${d.r})</div>
          <div class="text-xs mt-1">Agents: ${d.agent_count} / ${d.capacity}</div>
          <div class="text-xs">Habitability: ${(d.habitability * 100).toFixed(0)}%</div>
        `;
      })
      .on('mouseleave', () => {
        const tooltip = tooltipRef.current;
        if (tooltip) tooltip.style.display = 'none';
      });

    // Agent indicators
    hexes.each(function(d) {
      const group = d3.select(this);
      if (d.agent_count === 0) return;

      if (d.agent_count <= 5) {
        // Individual dots
        const dotRadius = 3;
        const positions = [
          [0, 0], [-6, -5], [6, -5], [-6, 5], [6, 5],
        ];
        d.agents.slice(0, 5).forEach((a, i) => {
          const [dx, dy] = positions[i];
          group.append('circle')
            .attr('cx', dx)
            .attr('cy', dy)
            .attr('r', dotRadius)
            .attr('fill', REGION_COLORS[a.processing_region] || '#6B7280')
            .attr('stroke', '#000')
            .attr('stroke-width', 0.5)
            .attr('pointer-events', 'none');
        });
      } else {
        // Count badge
        group.append('circle')
          .attr('cx', 0)
          .attr('cy', 0)
          .attr('r', 10)
          .attr('fill', 'rgba(0,0,0,0.7)')
          .attr('pointer-events', 'none');
        group.append('text')
          .attr('text-anchor', 'middle')
          .attr('dy', '0.35em')
          .attr('fill', '#fff')
          .attr('font-size', '9px')
          .attr('font-weight', 'bold')
          .attr('pointer-events', 'none')
          .text(d.agent_count);
      }
    });

  }, [gridData, colorMode, selectedTile]);

  if (!activeSessionId) {
    return <EmptyState message="Create or select a simulation session to view the hex map" />;
  }

  if (loading) {
    return (
      <div className="flex h-full items-center justify-center text-gray-500">
        Loading hex grid...
      </div>
    );
  }

  if (gridData && !gridData.enabled) {
    return (
      <div className="space-y-4">
        <h1 className="text-2xl font-bold text-gray-100">Hex Map</h1>
        <div className="rounded-lg border border-gray-800 bg-gray-900 p-8 text-center text-gray-500">
          Hex grid is not enabled. Enable <code className="text-gray-400">tick_config.enabled</code> and{' '}
          <code className="text-gray-400">hex_grid_config.enabled</code> in session config to use this view.
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col gap-4">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-gray-100">Hex Map</h1>
        <div className="flex items-center gap-2">
          {/* Color mode selector */}
          {(['terrain', 'density', 'habitability'] as ColorMode[]).map(mode => (
            <button
              key={mode}
              onClick={() => setColorMode(mode)}
              className={`rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
                colorMode === mode
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
              }`}
            >
              {mode.charAt(0).toUpperCase() + mode.slice(1)}
            </button>
          ))}
          <button
            onClick={loadGrid}
            className="ml-2 rounded-md bg-gray-800 px-3 py-1.5 text-xs text-gray-300 hover:bg-gray-700"
          >
            Refresh
          </button>
        </div>
      </div>

      {/* Stats cards */}
      {gridData?.stats && (
        <div className="grid grid-cols-4 gap-3">
          {[
            { label: 'Total Tiles', value: gridData.stats.total_tiles },
            { label: 'Habitable', value: gridData.stats.habitable_tiles },
            { label: 'Occupied', value: gridData.stats.occupied_tiles },
            { label: 'Agents on Grid', value: gridData.stats.total_agents_on_grid },
          ].map(({ label, value }) => (
            <div key={label} className="rounded-lg border border-gray-800 bg-gray-900 px-4 py-3">
              <div className="text-xs text-gray-500">{label}</div>
              <div className="mt-1 text-xl font-bold text-gray-100">{value}</div>
            </div>
          ))}
        </div>
      )}

      <div className="flex flex-1 gap-4 min-h-0">
        {/* Map */}
        <div className="relative flex-1 rounded-lg border border-gray-800 bg-gray-950 overflow-hidden">
          <svg ref={svgRef} className="w-full h-full" style={{ minHeight: 500 }} />
          {/* Tooltip */}
          <div
            ref={tooltipRef}
            className="pointer-events-none fixed z-50 rounded-lg border border-gray-700 bg-gray-900 px-3 py-2 text-sm text-gray-200 shadow-lg"
            style={{ display: 'none' }}
          />

          {/* Legend */}
          {colorMode === 'terrain' && (
            <div className="absolute bottom-3 left-3 rounded-lg border border-gray-800 bg-gray-900/90 p-2">
              <div className="text-xs font-semibold text-gray-400 mb-1">Terrain</div>
              <div className="grid grid-cols-2 gap-x-3 gap-y-0.5">
                {Object.entries(TERRAIN_COLORS).map(([key, color]) => (
                  <div key={key} className="flex items-center gap-1.5">
                    <div className="h-2.5 w-2.5 rounded-sm" style={{ backgroundColor: color }} />
                    <span className="text-[10px] text-gray-400">{TERRAIN_LABELS[key] || key}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Tile detail panel */}
        {selectedTile && (
          <div className="w-72 overflow-y-auto rounded-lg border border-gray-800 bg-gray-900 p-4">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-semibold text-gray-200">
                {TERRAIN_LABELS[selectedTile.terrain_type] || selectedTile.terrain_type}
              </h3>
              <button
                onClick={() => setSelectedTile(null)}
                className="text-xs text-gray-500 hover:text-gray-300"
              >
                Close
              </button>
            </div>
            <div className="space-y-2 text-xs text-gray-400">
              <div>Coordinates: ({selectedTile.q}, {selectedTile.r})</div>
              <div>Elevation: {selectedTile.elevation}m</div>
              <div>Habitability: {(selectedTile.habitability * 100).toFixed(0)}%</div>
              <div>Water: {(selectedTile.water_access * 100).toFixed(0)}%</div>
              <div>Soil: {(selectedTile.soil_quality * 100).toFixed(0)}%</div>
              <div>Resources: {(selectedTile.natural_resources * 100).toFixed(0)}%</div>
              <div>Vegetation: {(selectedTile.vegetation * 100).toFixed(0)}%</div>
              <div>Capacity: {selectedTile.agent_count} / {selectedTile.capacity}</div>
            </div>

            {/* Region counts */}
            {Object.keys(selectedTile.region_counts).length > 0 && (
              <div className="mt-3">
                <div className="text-xs font-semibold text-gray-400 mb-1">Processing Regions</div>
                {Object.entries(selectedTile.region_counts).map(([region, count]) => (
                  <div key={region} className="flex items-center gap-2 text-xs">
                    <div className="h-2 w-2 rounded-full" style={{ backgroundColor: REGION_COLORS[region] || '#666' }} />
                    <span className="text-gray-400">{region}: {count}</span>
                  </div>
                ))}
              </div>
            )}

            {/* Agent list */}
            {tileAgents.length > 0 && (
              <div className="mt-3">
                <div className="text-xs font-semibold text-gray-400 mb-1">Agents ({tileAgents.length})</div>
                <div className="max-h-48 space-y-1 overflow-y-auto">
                  {tileAgents.map(a => (
                    <div
                      key={a.id}
                      className="flex items-center justify-between rounded px-2 py-1 text-xs hover:bg-gray-800"
                    >
                      <span className="text-gray-300">{a.name}</span>
                      <span
                        className="rounded px-1.5 py-0.5 text-[10px]"
                        style={{
                          backgroundColor: (REGION_COLORS[a.processing_region] || '#666') + '33',
                          color: REGION_COLORS[a.processing_region] || '#666',
                        }}
                      >
                        {a.processing_region}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

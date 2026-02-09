import { useState, useEffect, useRef, useCallback } from 'react';
import { useSimulationStore } from '../../../stores/simulation';
import { EmptyState } from '../../shared/EmptyState';
import * as api from '../../../api/client';
import * as d3 from 'd3';
import type { NetworkGraph, NetworkNode } from '../../../types';
import { REGION_COLORS, REGION_LABELS, REGION_ORDER, EDGE_TYPE_COLORS } from '../../../lib/constants';

export function NetworkView() {
  const { activeSessionId, metrics } = useSimulationStore();
  const svgRef = useRef<SVGSVGElement>(null);
  const [graph, setGraph] = useState<NetworkGraph | null>(null);
  const [bondThreshold, setBondThreshold] = useState(0.1);
  const [showPartner, setShowPartner] = useState(true);
  const [showSocial, setShowSocial] = useState(true);
  const [showParent, setShowParent] = useState(true);
  const [selectedNode, setSelectedNode] = useState<NetworkNode | null>(null);

  const fetchGraph = useCallback(async () => {
    if (!activeSessionId) return;
    try {
      const data = await api.getNetworkGraph(activeSessionId, bondThreshold);
      setGraph(data);
    } catch {
      // ignore
    }
  }, [activeSessionId, bondThreshold]);

  useEffect(() => {
    fetchGraph();
  }, [fetchGraph, metrics.length]);

  // D3 force-directed graph
  useEffect(() => {
    if (!svgRef.current || !graph || graph.nodes.length === 0) return;

    const svg = d3.select(svgRef.current);
    const width = svgRef.current.clientWidth;
    const height = 500;

    svg.selectAll('*').remove();
    svg.attr('width', width).attr('height', height);

    // Filter edges by type
    const filteredEdges = graph.edges.filter((e) => {
      if (e.type === 'partner' && !showPartner) return false;
      if (e.type === 'social' && !showSocial) return false;
      if (e.type === 'parent' && !showParent) return false;
      return true;
    });

    // Connection counts for sizing
    const connCounts: Record<string, number> = {};
    for (const n of graph.nodes) connCounts[n.id] = 0;
    for (const e of filteredEdges) {
      connCounts[e.source] = (connCounts[e.source] || 0) + 1;
      connCounts[e.target] = (connCounts[e.target] || 0) + 1;
    }

    const sizeScale = d3.scaleSqrt()
      .domain([0, Math.max(...Object.values(connCounts), 1)])
      .range([4, 14]);

    // D3 simulation nodes/links (need mutable copies)
    type SimNode = NetworkNode & d3.SimulationNodeDatum;
    type SimLink = { source: string | SimNode; target: string | SimNode; type: string; strength: number };

    const simNodes: SimNode[] = graph.nodes.map((n) => ({ ...n }));
    const simLinks: SimLink[] = filteredEdges.map((e) => ({ ...e }));

    const simulation = d3.forceSimulation<SimNode>(simNodes)
      .force('link', d3.forceLink<SimNode, SimLink>(simLinks).id((d) => d.id).distance(60).strength(0.3))
      .force('charge', d3.forceManyBody().strength(-80))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide<SimNode>().radius((d) => sizeScale(connCounts[d.id] || 0) + 2));

    // Draw links
    const link = svg.append('g')
      .selectAll('line')
      .data(simLinks)
      .join('line')
      .attr('stroke', (d) => EDGE_TYPE_COLORS[d.type] || '#4B5563')
      .attr('stroke-width', (d) => d.type === 'partner' ? 3 : Math.max(1, d.strength * 3))
      .attr('stroke-dasharray', (d) => d.type === 'parent' ? '4,4' : 'none')
      .attr('stroke-opacity', 0.6);

    // Draw nodes
    const node = svg.append('g')
      .selectAll('circle')
      .data(simNodes)
      .join('circle')
      .attr('r', (d) => sizeScale(connCounts[d.id] || 0))
      .attr('fill', (d) => REGION_COLORS[d.region] || '#6B7280')
      .attr('stroke', '#111827')
      .attr('stroke-width', 1)
      .attr('cursor', 'pointer')
      .on('click', (_event, d) => {
        setSelectedNode(d as NetworkNode);
      })
      .call(
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        d3.drag<any, SimNode>()
          .on('start', (event, d) => {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
          })
          .on('drag', (event, d) => {
            d.fx = event.x;
            d.fy = event.y;
          })
          .on('end', (event, d) => {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = undefined;
            d.fy = undefined;
          }) as any
      );

    node.append('title')
      .text((d) => `${d.name}\n${d.region}\n${connCounts[d.id] || 0} connections`);

    simulation.on('tick', () => {
      link
        .attr('x1', (d) => ((d.source as SimNode).x ?? 0))
        .attr('y1', (d) => ((d.source as SimNode).y ?? 0))
        .attr('x2', (d) => ((d.target as SimNode).x ?? 0))
        .attr('y2', (d) => ((d.target as SimNode).y ?? 0));

      node
        .attr('cx', (d) => (d.x = Math.max(10, Math.min(width - 10, d.x ?? 0))))
        .attr('cy', (d) => (d.y = Math.max(10, Math.min(height - 10, d.y ?? 0))));
    });

    return () => {
      simulation.stop();
    };
  }, [graph, showPartner, showSocial, showParent]);

  if (!activeSessionId || metrics.length === 0) {
    return <EmptyState message="Run a simulation to see the social network" />;
  }

  if (!graph) {
    return <EmptyState message="Loading network data..." />;
  }

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-gray-100">Social Network</h1>

      {/* Controls */}
      <div className="flex flex-wrap items-center gap-6 rounded-lg border border-gray-800 bg-gray-900 p-4">
        <div className="flex items-center gap-2">
          <label className="text-sm text-gray-400">Bond Threshold</label>
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={bondThreshold}
            onChange={(e) => setBondThreshold(Number(e.target.value))}
            className="w-24 accent-blue-600"
          />
          <span className="font-mono text-sm text-gray-300">{bondThreshold.toFixed(2)}</span>
        </div>
        <label className="flex items-center gap-2 text-sm text-gray-400 cursor-pointer">
          <input type="checkbox" checked={showPartner} onChange={(e) => setShowPartner(e.target.checked)} className="accent-blue-600" />
          <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ backgroundColor: EDGE_TYPE_COLORS.partner }} />
          Partners
        </label>
        <label className="flex items-center gap-2 text-sm text-gray-400 cursor-pointer">
          <input type="checkbox" checked={showSocial} onChange={(e) => setShowSocial(e.target.checked)} className="accent-blue-600" />
          <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ backgroundColor: EDGE_TYPE_COLORS.social }} />
          Social Bonds
        </label>
        <label className="flex items-center gap-2 text-sm text-gray-400 cursor-pointer">
          <input type="checkbox" checked={showParent} onChange={(e) => setShowParent(e.target.checked)} className="accent-blue-600" />
          <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ backgroundColor: EDGE_TYPE_COLORS.parent }} />
          Parent Links
        </label>
      </div>

      {/* Legend */}
      <div className="flex flex-wrap gap-4">
        {REGION_ORDER.map((region) => (
          <div key={region} className="flex items-center gap-2">
            <div className="h-3 w-3 rounded-full" style={{ backgroundColor: REGION_COLORS[region] }} />
            <span className="text-sm text-gray-400">{REGION_LABELS[region]}</span>
          </div>
        ))}
      </div>

      {/* Graph */}
      <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
        <svg ref={svgRef} className="w-full" style={{ minHeight: 500 }} />
      </div>

      {/* Stats */}
      <div className="grid grid-cols-4 gap-4">
        <div className="rounded-lg border border-gray-800 bg-gray-900 p-4 text-center">
          <div className="text-2xl font-bold text-gray-100">{graph.stats.total_nodes}</div>
          <div className="text-sm text-gray-400">Nodes</div>
        </div>
        <div className="rounded-lg border border-gray-800 bg-gray-900 p-4 text-center">
          <div className="text-2xl font-bold text-gray-100">{graph.stats.total_edges}</div>
          <div className="text-sm text-gray-400">Edges</div>
        </div>
        <div className="rounded-lg border border-gray-800 bg-gray-900 p-4 text-center">
          <div className="text-2xl font-bold text-gray-100">{graph.stats.avg_connections}</div>
          <div className="text-sm text-gray-400">Avg Connections</div>
        </div>
        <div className="rounded-lg border border-gray-800 bg-gray-900 p-4 text-center">
          <div className="text-2xl font-bold text-gray-100">{graph.stats.connected_components}</div>
          <div className="text-sm text-gray-400">Components</div>
        </div>
      </div>

      {/* Selected node sidebar */}
      {selectedNode && (
        <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold text-gray-200">
              {selectedNode.name}
              <span className="ml-2 text-sm text-gray-500">({selectedNode.region})</span>
            </h3>
            <button
              onClick={() => setSelectedNode(null)}
              className="text-sm text-gray-500 hover:text-gray-300"
            >
              Clear
            </button>
          </div>
          <div className="mt-2 grid grid-cols-3 gap-4 text-sm">
            <div><span className="text-gray-500">ID:</span> <span className="font-mono text-gray-200">{selectedNode.id}</span></div>
            <div><span className="text-gray-500">Region:</span> <span className="text-gray-200">{selectedNode.region}</span></div>
            <div><span className="text-gray-500">Location:</span> <span className="text-gray-200">{selectedNode.location_id || 'none'}</span></div>
          </div>
          <div className="mt-3 text-sm text-gray-400">
            Connections: {graph.edges.filter((e) => e.source === selectedNode.id || e.target === selectedNode.id).length}
          </div>
        </div>
      )}
    </div>
  );
}

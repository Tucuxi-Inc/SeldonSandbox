import { useState, useEffect, useRef } from 'react';
import { useSimulationStore } from '../../../stores/simulation';
import { EmptyState } from '../../shared/EmptyState';
import { REGION_COLORS } from '../../../lib/constants';
import * as api from '../../../api/client';
import type { FamilyTree, FamilyTreeNode } from '../../../types';
import * as d3 from 'd3';

export function LineageView() {
  const { activeSessionId, agents } = useSimulationStore();
  const [selectedAgentId, setSelectedAgentId] = useState('');
  const [familyTree, setFamilyTree] = useState<FamilyTree | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  const handleLoad = async () => {
    if (!activeSessionId || !selectedAgentId) return;
    const tree = await api.getFamilyTree(activeSessionId, selectedAgentId, 3, 3);
    setFamilyTree(tree);
  };

  useEffect(() => {
    if (!familyTree?.root || !svgRef.current) return;

    const svg = d3.select(svgRef.current);
    const width = svgRef.current.clientWidth;
    const height = 600;
    svg.selectAll('*').remove();
    svg.attr('width', width).attr('height', height);

    // Build tree data structure
    interface TreeNode {
      name: string;
      id: string;
      region: string;
      isOutsider: boolean;
      birthOrder: number;
      isAlive: boolean;
      children: TreeNode[];
    }

    const rootNode: TreeNode = {
      name: familyTree.root.name,
      id: familyTree.root.id,
      region: familyTree.root.processing_region,
      isOutsider: familyTree.root.is_outsider,
      birthOrder: familyTree.root.birth_order,
      isAlive: familyTree.root.is_alive,
      children: (familyTree.descendants || []).map(function mapDesc(d: FamilyTreeNode): TreeNode {
        return {
          name: d.name,
          id: d.id,
          region: d.processing_region,
          isOutsider: d.is_outsider,
          birthOrder: d.birth_order,
          isAlive: d.is_alive,
          children: (d.children || []).map(mapDesc),
        };
      }),
    };

    const root = d3.hierarchy(rootNode);
    const treeLayout = d3.tree<TreeNode>().size([width - 100, height - 100]);
    treeLayout(root);

    const g = svg.append('g').attr('transform', 'translate(50, 50)');

    // Links
    g.selectAll('.link')
      .data(root.links())
      .join('path')
      .attr('class', 'link')
      .attr('d', d3.linkVertical<d3.HierarchyPointLink<TreeNode>, d3.HierarchyPointNode<TreeNode>>()
        .x((d) => d.x)
        .y((d) => d.y) as unknown as string)
      .attr('fill', 'none')
      .attr('stroke', '#374151')
      .attr('stroke-width', 1.5);

    // Nodes
    const nodes = g.selectAll('.node')
      .data(root.descendants())
      .join('g')
      .attr('class', 'node')
      .attr('transform', (d) => `translate(${d.x}, ${d.y})`);

    // Node circles
    nodes.append('circle')
      .attr('r', 14)
      .attr('fill', (d) => REGION_COLORS[d.data.region] || '#6B7280')
      .attr('stroke', (d) => d.data.isOutsider ? '#F59E0B' : d.data.isAlive ? '#fff' : '#4b5563')
      .attr('stroke-width', (d) => d.data.isOutsider ? 3 : 2)
      .attr('stroke-dasharray', (d) => d.data.isOutsider ? '4,2' : 'none')
      .attr('opacity', (d) => d.data.isAlive ? 1 : 0.5);

    // Birth order number
    nodes.append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', '0.35em')
      .attr('fill', '#fff')
      .attr('font-size', '10px')
      .attr('font-weight', 'bold')
      .text((d) => d.data.birthOrder);

    // Name label
    nodes.append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', 28)
      .attr('fill', '#9ca3af')
      .attr('font-size', '10px')
      .text((d) => d.data.name);

    // Ancestor nodes above root
    if (familyTree.ancestors.length > 0) {
      const ancestorY = -40;
      const ancestorSpacing = 80;
      const startX = (width - 100) / 2 - ((familyTree.ancestors.length - 1) * ancestorSpacing) / 2;

      familyTree.ancestors.forEach((anc, i) => {
        const x = startX + i * ancestorSpacing;
        const ag = g.append('g').attr('transform', `translate(${x}, ${ancestorY})`);

        // Line from ancestor to root
        g.append('line')
          .attr('x1', x).attr('y1', ancestorY + 14)
          .attr('x2', root.x!).attr('y2', root.y! - 14)
          .attr('stroke', '#374151').attr('stroke-width', 1.5);

        ag.append('circle')
          .attr('r', 12)
          .attr('fill', REGION_COLORS[anc.processing_region] || '#6B7280')
          .attr('stroke', anc.is_outsider ? '#F59E0B' : '#fff')
          .attr('stroke-width', 2)
          .attr('opacity', anc.is_alive ? 1 : 0.5);

        ag.append('text')
          .attr('text-anchor', 'middle')
          .attr('dy', -18)
          .attr('fill', '#9ca3af')
          .attr('font-size', '9px')
          .text(anc.name);
      });
    }
  }, [familyTree]);

  if (agents.length === 0) {
    return <EmptyState message="Run a simulation to explore lineages" />;
  }

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-gray-100">Family & Lineage</h1>

      {/* Agent Selector */}
      <div className="flex items-center gap-4">
        <select
          value={selectedAgentId}
          onChange={(e) => setSelectedAgentId(e.target.value)}
          className="rounded-md border border-gray-700 bg-gray-900 px-3 py-2 text-sm text-gray-200"
        >
          <option value="">Select an agent...</option>
          {agents.map((a) => (
            <option key={a.id} value={a.id}>{a.name} ({a.id})</option>
          ))}
        </select>
        <button
          onClick={handleLoad}
          disabled={!selectedAgentId}
          className="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-40"
        >
          Load Family Tree
        </button>
      </div>

      {/* Legend */}
      <div className="flex flex-wrap gap-4 text-sm">
        <div className="flex items-center gap-2">
          <div className="h-4 w-4 rounded-full border-2 border-dashed border-amber-500 bg-gray-700" />
          <span className="text-gray-400">Outsider lineage</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="h-4 w-4 rounded-full bg-gray-700 opacity-50" />
          <span className="text-gray-400">Deceased</span>
        </div>
        <div className="text-gray-500">Number in circle = birth order</div>
      </div>

      {/* Tree */}
      {familyTree ? (
        <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
          <svg ref={svgRef} className="w-full" style={{ minHeight: 600 }} />
        </div>
      ) : (
        <EmptyState message="Select an agent and load their family tree" />
      )}
    </div>
  );
}

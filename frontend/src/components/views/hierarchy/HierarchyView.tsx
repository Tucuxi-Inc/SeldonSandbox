import { useState, useEffect } from 'react';
import { useSimulationStore } from '../../../stores/simulation';
import * as api from '../../../api/client';
import type { HierarchyOverview, RoleBreakdown, InfluenceEntry, MentorshipChain } from '../../../types';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell,
} from 'recharts';

const ROLE_COLORS: Record<string, string> = {
  leader: '#f59e0b',
  innovator: '#8b5cf6',
  mediator: '#22c55e',
  worker: '#3b82f6',
  outsider_bridge: '#ec4899',
  unassigned: '#6b7280',
};

export function HierarchyView() {
  const { activeSessionId } = useSimulationStore();
  const [hierarchy, setHierarchy] = useState<HierarchyOverview | null>(null);
  const [roles, setRoles] = useState<RoleBreakdown | null>(null);
  const [influence, setInfluence] = useState<InfluenceEntry[]>([]);
  const [mentorship, setMentorship] = useState<MentorshipChain[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!activeSessionId) return;
    setLoading(true);
    Promise.all([
      api.getHierarchy(activeSessionId).catch(() => null),
      api.getRoles(activeSessionId).catch(() => null),
      api.getInfluenceMap(activeSessionId).catch(() => ({ enabled: false, agents: [] })),
      api.getMentorship(activeSessionId).catch(() => ({ enabled: false, chains: [] })),
    ]).then(([h, r, inf, m]) => {
      setHierarchy(h);
      setRoles(r);
      setInfluence(inf?.agents ?? []);
      setMentorship(m?.chains ?? []);
    }).finally(() => setLoading(false));
  }, [activeSessionId]);

  if (!activeSessionId) {
    return <div className="p-6 text-gray-400">Create a session to view hierarchy.</div>;
  }

  if (loading) return <div className="p-6 text-gray-400">Loading...</div>;
  if (!hierarchy?.enabled) {
    return <div className="p-6 text-gray-400">Social dynamics extension not enabled. Enable it in Mission Control.</div>;
  }

  const statusData = hierarchy.status_distribution
    ? Object.entries(hierarchy.status_distribution).map(([bucket, count]) => ({ bucket, count }))
    : [];

  const roleData = roles?.roles
    ? Object.entries(roles.roles).map(([name, count]) => ({ name, value: count }))
    : [];

  const topInfluencers = influence.slice(0, 20);

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-gray-100">Social Hierarchy</h1>

      {/* Summary */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        <div className="rounded-lg border border-gray-800 bg-gray-900 p-4 text-center">
          <div className="text-sm text-gray-500">Mean Status</div>
          <div className="text-2xl font-bold text-gray-100">{hierarchy.mean_status?.toFixed(2) ?? 0}</div>
        </div>
        <div className="rounded-lg border border-gray-800 bg-gray-900 p-4 text-center">
          <div className="text-sm text-gray-500">Total Roles</div>
          <div className="text-2xl font-bold text-gray-100">{roleData.length}</div>
        </div>
        <div className="rounded-lg border border-gray-800 bg-gray-900 p-4 text-center">
          <div className="text-sm text-gray-500">Mentorship Chains</div>
          <div className="text-2xl font-bold text-gray-100">{mentorship.length}</div>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Status Distribution */}
        {statusData.length > 0 && (
          <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
            <h2 className="mb-3 text-lg font-semibold text-gray-200">Status Distribution</h2>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={statusData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="bucket" stroke="#9ca3af" tick={{ fontSize: 11 }} />
                <YAxis stroke="#9ca3af" tick={{ fontSize: 11 }} />
                <Tooltip contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }} />
                <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Role Breakdown */}
        {roleData.length > 0 && (
          <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
            <h2 className="mb-3 text-lg font-semibold text-gray-200">Role Breakdown</h2>
            <ResponsiveContainer width="100%" height={250}>
              <PieChart>
                <Pie
                  data={roleData}
                  dataKey="value"
                  nameKey="name"
                  cx="50%"
                  cy="50%"
                  outerRadius={90}
                  label={({ name, percent }: any) => `${name} ${(percent * 100).toFixed(0)}%`}
                >
                  {roleData.map((entry) => (
                    <Cell key={entry.name} fill={ROLE_COLORS[entry.name] ?? '#6b7280'} />
                  ))}
                </Pie>
                <Tooltip contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Top Influencers */}
        <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
          <h2 className="mb-3 text-lg font-semibold text-gray-200">Top Influencers</h2>
          {topInfluencers.length === 0 ? (
            <div className="text-sm text-gray-500">No influence data.</div>
          ) : (
            <div className="max-h-60 overflow-y-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-800 text-left text-gray-500">
                    <th className="pb-2">Agent</th>
                    <th className="pb-2">Role</th>
                    <th className="pb-2">Influence</th>
                  </tr>
                </thead>
                <tbody>
                  {topInfluencers.map((a) => (
                    <tr key={a.agent_id} className="border-b border-gray-800/50">
                      <td className="py-1.5 text-gray-300">{a.agent_name}</td>
                      <td className="py-1.5">
                        <span
                          className="rounded px-1.5 py-0.5 text-xs"
                          style={{ backgroundColor: (ROLE_COLORS[a.social_role ?? ''] ?? '#6b7280') + '33', color: ROLE_COLORS[a.social_role ?? ''] ?? '#9ca3af' }}
                        >
                          {a.social_role ?? 'unassigned'}
                        </span>
                      </td>
                      <td className="py-1.5 font-mono text-gray-300">{a.influence_score.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Mentorship Chains */}
        <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
          <h2 className="mb-3 text-lg font-semibold text-gray-200">Mentorship Chains</h2>
          {mentorship.length === 0 ? (
            <div className="text-sm text-gray-500">No active mentorship chains.</div>
          ) : (
            <div className="max-h-60 overflow-y-auto space-y-2">
              {mentorship.map((chain) => (
                <div key={chain.mentor_id} className="rounded border border-gray-800 bg-gray-950 p-2">
                  <div className="text-sm font-medium text-amber-400">{chain.mentor_name}</div>
                  <div className="mt-1 flex flex-wrap gap-1">
                    {chain.mentees.map((m) => (
                      <span key={m.id} className="rounded bg-gray-800 px-1.5 py-0.5 text-xs text-gray-400">
                        {m.name}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

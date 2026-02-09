import { useState, useEffect } from 'react';
import { useSimulationStore } from '../../../stores/simulation';
import * as api from '../../../api/client';
import type { AlleleFrequencies, EpigeneticPrevalence, TraitGeneCorrelation } from '../../../types';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from 'recharts';

export function GeneticsView() {
  const { activeSessionId } = useSimulationStore();
  const [alleles, setAlleles] = useState<AlleleFrequencies | null>(null);
  const [epigenetics, setEpigenetics] = useState<EpigeneticPrevalence | null>(null);
  const [correlations, setCorrelations] = useState<TraitGeneCorrelation | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!activeSessionId) return;
    setLoading(true);
    Promise.all([
      api.getAlleleFrequencies(activeSessionId).catch(() => null),
      api.getEpigeneticPrevalence(activeSessionId).catch(() => null),
      api.getTraitGeneCorrelation(activeSessionId).catch(() => null),
    ]).then(([a, e, c]) => {
      setAlleles(a);
      setEpigenetics(e);
      setCorrelations(c);
    }).finally(() => setLoading(false));
  }, [activeSessionId]);

  if (!activeSessionId) {
    return <div className="p-6 text-gray-400">Create a session to view genetics.</div>;
  }

  if (loading) return <div className="p-6 text-gray-400">Loading...</div>;
  if (!alleles?.enabled && !epigenetics?.enabled) {
    return <div className="p-6 text-gray-400">Genetics not enabled. Enable genetics in the Genetics & Epigenetics config section on Mission Control.</div>;
  }

  const alleleData = alleles?.loci
    ? Object.entries(alleles.loci).map(([locus, info]) => ({
        locus,
        trait: info.trait,
        dominant: info.dominant_frequency,
        recessive: info.recessive_frequency,
      }))
    : [];

  const epiData = epigenetics?.markers
    ? Object.entries(epigenetics.markers).map(([name, info]) => ({
        marker: name.replace(/_/g, ' '),
        prevalence: Math.round(info.prevalence * 100),
        active: info.active_count,
        total: info.total,
      }))
    : [];

  const corrData = correlations?.correlations
    ? Object.entries(correlations.correlations).map(([locus, corr]) => ({
        locus,
        correlation: corr,
        absCorr: Math.abs(corr),
      }))
    : [];

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-gray-100">Genetics & Epigenetics</h1>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Allele Frequencies */}
        {alleleData.length > 0 && (
          <div className="rounded-lg border border-gray-800 bg-gray-900 p-4 lg:col-span-2">
            <h2 className="mb-3 text-lg font-semibold text-gray-200">Allele Frequencies by Locus</h2>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={alleleData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="locus" stroke="#9ca3af" tick={{ fontSize: 10 }} angle={-30} textAnchor="end" height={50} />
                <YAxis stroke="#9ca3af" tick={{ fontSize: 11 }} domain={[0, 1]} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }}
                  formatter={((value: any) => [Number(value).toFixed(3)]) as any}
                />
                <Legend />
                <Bar dataKey="dominant" name="Dominant (A)" fill="#22c55e" stackId="a" />
                <Bar dataKey="recessive" name="Recessive (a)" fill="#ef4444" stackId="a" />
              </BarChart>
            </ResponsiveContainer>
            <div className="mt-2 flex flex-wrap gap-2">
              {alleleData.map((d) => (
                <span key={d.locus} className="rounded bg-gray-800 px-2 py-0.5 text-xs text-gray-400">
                  {d.locus}: {d.trait}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Epigenetic Marker Prevalence */}
        {epiData.length > 0 && (
          <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
            <h2 className="mb-3 text-lg font-semibold text-gray-200">Epigenetic Markers</h2>
            <div className="space-y-3">
              {epiData.map((m) => (
                <div key={m.marker}>
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-gray-400">{m.marker}</span>
                    <span className="font-mono text-gray-300">{m.prevalence}%</span>
                  </div>
                  <div className="mt-1 h-2 rounded-full bg-gray-800">
                    <div
                      className="h-2 rounded-full bg-purple-500 transition-all"
                      style={{ width: `${m.prevalence}%` }}
                    />
                  </div>
                  <div className="text-xs text-gray-600">{m.active}/{m.total} active</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Trait-Gene Correlations */}
        {corrData.length > 0 && (
          <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
            <h2 className="mb-3 text-lg font-semibold text-gray-200">Trait-Gene Correlations</h2>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={corrData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis type="number" stroke="#9ca3af" tick={{ fontSize: 11 }} domain={[-1, 1]} />
                <YAxis type="category" dataKey="locus" stroke="#9ca3af" tick={{ fontSize: 10 }} width={60} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }}
                  formatter={((value: any) => [Number(value).toFixed(3), 'Correlation']) as any}
                />
                <Bar dataKey="correlation" fill="#3b82f6">
                  {corrData.map((entry, i) => (
                    <rect key={i} fill={entry.correlation >= 0 ? '#22c55e' : '#ef4444'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>
    </div>
  );
}

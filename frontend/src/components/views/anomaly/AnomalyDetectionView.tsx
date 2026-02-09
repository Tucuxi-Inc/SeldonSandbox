import { useState, useEffect } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
} from 'recharts';
import { useSimulationStore } from '../../../stores/simulation';
import { EmptyState } from '../../shared/EmptyState';
import * as api from '../../../api/client';
import type { AnomalyReport, Anomaly } from '../../../types';
import { ANOMALY_COLORS } from '../../../lib/constants';

function severityColor(score: number): string {
  if (score >= 3.0) return ANOMALY_COLORS.critical;
  if (score >= 2.5) return ANOMALY_COLORS.high;
  if (score >= 2.0) return ANOMALY_COLORS.medium;
  if (score >= 1.5) return ANOMALY_COLORS.warning;
  return ANOMALY_COLORS.normal;
}

function SeverityBadge({ severity }: { severity: string }) {
  const colors: Record<string, string> = {
    critical: 'bg-red-900 text-red-200',
    high: 'bg-orange-900 text-orange-200',
    medium: 'bg-yellow-900 text-yellow-200',
  };
  return (
    <span className={`rounded px-2 py-0.5 text-xs font-medium ${colors[severity] || 'bg-gray-800 text-gray-300'}`}>
      {severity}
    </span>
  );
}

export function AnomalyDetectionView() {
  const { activeSessionId, metrics } = useSimulationStore();
  const [report, setReport] = useState<AnomalyReport | null>(null);
  const [selectedAnomaly, setSelectedAnomaly] = useState<Anomaly | null>(null);

  useEffect(() => {
    if (!activeSessionId) return;
    api.getAnomalies(activeSessionId).then(setReport).catch(() => {});
  }, [activeSessionId, metrics.length]);

  if (!activeSessionId || metrics.length === 0) {
    return <EmptyState message="Run a simulation to detect anomalies" />;
  }

  if (!report) {
    return <EmptyState message="Loading anomaly data..." />;
  }

  const chartData = report.generation_scores.map((score, i) => ({
    generation: i,
    score: Math.round(score * 100) / 100,
  }));

  // Build overlay data for selected anomaly's metric
  const overlayData = selectedAnomaly
    ? metrics.map((m, i) => ({
        generation: i,
        value: (m as unknown as Record<string, unknown>)[selectedAnomaly.metric] as number ?? 0,
      }))
    : null;

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-gray-100">Anomaly Detection</h1>
      <p className="text-sm text-gray-400">
        Statistical outliers detected via z-score analysis across {metrics.length} generations.
        {report.anomalies.length === 0 && ' No anomalies found.'}
      </p>

      {/* Anomaly Score Timeline */}
      <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
        <h2 className="mb-4 text-lg font-semibold text-gray-200">Anomaly Score by Generation</h2>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="generation" stroke="#6b7280" fontSize={11} />
            <YAxis stroke="#6b7280" fontSize={11} />
            <Tooltip
              contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: '8px' }}
              labelStyle={{ color: '#9ca3af' }}
              itemStyle={{ color: '#e5e7eb' }}
              formatter={((value: number | undefined) => [(value ?? 0).toFixed(2), 'Max Z-Score']) as any}
            />
            <Bar dataKey="score" radius={[2, 2, 0, 0]}>
              {chartData.map((entry) => (
                <Cell key={entry.generation} fill={severityColor(entry.score)} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Metric overlay when anomaly selected */}
      {selectedAnomaly && overlayData && (
        <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-gray-200">
              {selectedAnomaly.metric} over time
            </h2>
            <button
              onClick={() => setSelectedAnomaly(null)}
              className="text-sm text-gray-500 hover:text-gray-300"
            >
              Clear
            </button>
          </div>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={overlayData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="generation" stroke="#6b7280" fontSize={11} />
              <YAxis stroke="#6b7280" fontSize={11} />
              <Tooltip
                contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: '8px' }}
                labelStyle={{ color: '#9ca3af' }}
                itemStyle={{ color: '#e5e7eb' }}
              />
              <Bar dataKey="value" fill="#3B82F6" radius={[2, 2, 0, 0]}>
                {overlayData.map((entry) => (
                  <Cell
                    key={entry.generation}
                    fill={entry.generation === selectedAnomaly.generation ? '#EF4444' : '#3B82F6'}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Anomaly Cards */}
      {report.anomalies.length > 0 && (
        <div className="space-y-3">
          <h2 className="text-lg font-semibold text-gray-200">
            Detected Anomalies ({report.anomalies.length})
          </h2>
          {report.anomalies.map((anomaly, i) => (
            <div
              key={i}
              onClick={() => setSelectedAnomaly(anomaly)}
              className={`cursor-pointer rounded-lg border p-4 transition-colors ${
                selectedAnomaly === anomaly
                  ? 'border-blue-600 bg-gray-800'
                  : 'border-gray-800 bg-gray-900 hover:border-gray-700'
              }`}
            >
              <div className="flex items-center gap-3">
                <SeverityBadge severity={anomaly.severity} />
                <span className="rounded bg-gray-800 px-2 py-0.5 text-xs text-gray-400">
                  {anomaly.category}
                </span>
                <span className="text-sm text-gray-400">Gen {anomaly.generation}</span>
                <span className="ml-auto font-mono text-sm text-gray-300">
                  z={anomaly.z_score.toFixed(2)}
                </span>
              </div>
              <p className="mt-2 text-sm text-gray-300">{anomaly.description}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

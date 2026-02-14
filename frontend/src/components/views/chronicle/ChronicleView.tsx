import { useState, useEffect } from 'react';
import { useSimulationStore } from '../../../stores/simulation';
import { EmptyState } from '../../shared/EmptyState';
import { SEVERITY_COLORS } from '../../../lib/constants';
import * as api from '../../../api/client';
import type { ChronicleIndex, ChronicleEntry, NotableEvent } from '../../../types';

const SEVERITY_LABELS: Record<string, string> = {
  minor: 'Minor',
  notable: 'Notable',
  major: 'Major',
  critical: 'Critical',
};

const SEVERITY_ORDER: Record<string, number> = {
  minor: 0,
  notable: 1,
  major: 2,
  critical: 3,
};

export function ChronicleView() {
  const { activeSessionId } = useSimulationStore();

  const [index, setIndex] = useState<ChronicleIndex | null>(null);
  const [selectedGeneration, setSelectedGeneration] = useState<number | null>(null);
  const [entry, setEntry] = useState<ChronicleEntry | null>(null);
  const [loadingIndex, setLoadingIndex] = useState(false);
  const [loadingEntry, setLoadingEntry] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch the chronicle index when the session changes
  useEffect(() => {
    if (!activeSessionId) {
      setIndex(null);
      setSelectedGeneration(null);
      setEntry(null);
      return;
    }

    setLoadingIndex(true);
    setError(null);
    api.getChronicleIndex(activeSessionId)
      .then((data) => {
        setIndex(data);
        // Auto-select the latest generation if available
        if (data.generations.length > 0) {
          const latest = data.generations[data.generations.length - 1];
          setSelectedGeneration(latest.generation);
        } else {
          setSelectedGeneration(null);
          setEntry(null);
        }
      })
      .catch(() => {
        setError('Failed to load chronicle index. The chronicle API may not be available yet.');
        setIndex(null);
      })
      .finally(() => setLoadingIndex(false));
  }, [activeSessionId]);

  // Fetch the chronicle entry when the selected generation changes
  useEffect(() => {
    if (!activeSessionId || selectedGeneration === null) {
      setEntry(null);
      return;
    }

    setLoadingEntry(true);
    api.getChronicle(activeSessionId, selectedGeneration)
      .then((data) => {
        setEntry(data);
      })
      .catch(() => {
        setEntry(null);
      })
      .finally(() => setLoadingEntry(false));
  }, [activeSessionId, selectedGeneration]);

  // --- No session state ---
  if (!activeSessionId) {
    return (
      <div className="p-6">
        <EmptyState message="Create or select a session to read The Seldon Chronicle." />
      </div>
    );
  }

  // --- Loading index ---
  if (loadingIndex) {
    return <div className="p-6 text-gray-400">Loading chronicle index...</div>;
  }

  // --- Error state ---
  if (error) {
    return (
      <div className="p-6">
        <div className="rounded-lg border border-gray-800 bg-gray-900 p-6">
          <h2 className="text-lg font-semibold text-gray-200">The Seldon Chronicle</h2>
          <p className="mt-2 text-gray-400">{error}</p>
        </div>
      </div>
    );
  }

  // --- Empty index ---
  if (!index || index.generations.length === 0) {
    return (
      <div className="p-6">
        <EmptyState message="No chronicle entries yet. Run the simulation to generate history." />
      </div>
    );
  }

  const generations = index.generations;

  // Sort events by severity (critical first) for the newspaper layout
  const sortedEvents = entry
    ? [...entry.events].sort(
        (a, b) => (SEVERITY_ORDER[b.severity] ?? 0) - (SEVERITY_ORDER[a.severity] ?? 0)
      )
    : [];

  return (
    <div className="flex h-full gap-0">
      {/* Left sidebar: Generation index */}
      <div className="flex w-64 shrink-0 flex-col border-r border-gray-800 bg-gray-950">
        <div className="border-b border-gray-800 px-4 py-3">
          <h2 className="text-sm font-semibold uppercase tracking-wider text-gray-400">
            The Seldon Chronicle
          </h2>
          <p className="mt-0.5 text-xs text-gray-500">{generations.length} issues</p>
        </div>
        <div className="flex-1 overflow-y-auto">
          {generations.map((gen) => {
            const isSelected = gen.generation === selectedGeneration;
            const severityColor = SEVERITY_COLORS[gen.max_severity] ?? SEVERITY_COLORS.minor;
            return (
              <button
                key={gen.generation}
                onClick={() => setSelectedGeneration(gen.generation)}
                className={`flex w-full items-center gap-3 border-b border-gray-800/50 px-4 py-2.5 text-left transition-colors ${
                  isSelected
                    ? 'bg-gray-800 text-gray-100'
                    : 'text-gray-400 hover:bg-gray-900 hover:text-gray-200'
                }`}
              >
                <span
                  className="inline-block h-2.5 w-2.5 shrink-0 rounded-full"
                  style={{ backgroundColor: severityColor }}
                  title={`Max severity: ${gen.max_severity}`}
                />
                <span className="flex-1">
                  <span className="block text-sm font-medium">Gen {gen.generation}</span>
                  <span className="block text-xs text-gray-500">
                    Pop {gen.population_size}
                  </span>
                </span>
                <span className="shrink-0 rounded bg-gray-700/50 px-1.5 py-0.5 text-xs font-mono text-gray-400">
                  {gen.event_count}
                </span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Right content: Newspaper issue */}
      <div className="flex-1 overflow-y-auto bg-gray-950 p-6">
        {loadingEntry ? (
          <div className="text-gray-400">Loading generation chronicle...</div>
        ) : entry ? (
          <div className="mx-auto max-w-3xl">
            {/* Issue header */}
            <div className="mb-6 border-b-2 border-gray-700 pb-4">
              <h1 className="text-3xl font-bold tracking-tight text-gray-100">
                Generation {entry.generation}
              </h1>
              <div className="mt-2 flex items-center gap-4 text-sm text-gray-400">
                <span>
                  Population: <span className="font-semibold text-gray-200">{entry.population_size}</span>
                </span>
                <span className="text-gray-700">|</span>
                <span>
                  Births: <span className="font-semibold text-green-400">{entry.births}</span>
                </span>
                <span className="text-gray-700">|</span>
                <span>
                  Deaths: <span className="font-semibold text-red-400">{entry.deaths}</span>
                </span>
              </div>
            </div>

            {/* Event cards */}
            {sortedEvents.length === 0 ? (
              <p className="text-gray-500">No notable events recorded for this generation.</p>
            ) : (
              <div className="space-y-4">
                {sortedEvents.map((event, i) => (
                  <EventCard key={i} event={event} agentNames={entry.agent_names ?? {}} />
                ))}
              </div>
            )}
          </div>
        ) : (
          <div className="text-gray-500">Select a generation from the index to view its chronicle.</div>
        )}
      </div>
    </div>
  );
}

function EventCard({ event, agentNames }: { event: NotableEvent; agentNames: Record<string, string> }) {
  const severityColor = SEVERITY_COLORS[event.severity] ?? SEVERITY_COLORS.minor;
  const severityLabel = SEVERITY_LABELS[event.severity] ?? event.severity;

  return (
    <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1">
          {/* Severity badge + event type */}
          <div className="mb-2 flex items-center gap-2">
            <span
              className="inline-flex items-center rounded px-2 py-0.5 text-xs font-semibold"
              style={{
                backgroundColor: severityColor + '20',
                color: severityColor,
                border: `1px solid ${severityColor}40`,
              }}
            >
              {severityLabel}
            </span>
            <span className="text-xs text-gray-500">
              {formatEventType(event.event_type)}
            </span>
          </div>

          {/* Headline */}
          <h3 className="text-base font-semibold text-gray-100">{event.headline}</h3>

          {/* Detail */}
          {event.detail && (
            <p className="mt-1.5 text-sm leading-relaxed text-gray-400">{event.detail}</p>
          )}

          {/* Agent references as named pills */}
          {event.agent_ids.length > 0 && (
            <div className="mt-2.5 flex flex-wrap gap-1.5">
              {event.agent_ids.map((agentId) => {
                const name = agentNames[agentId];
                return (
                  <span
                    key={agentId}
                    className="inline-flex items-center gap-1 rounded-full bg-gray-800 px-2.5 py-1 text-xs text-gray-300 border border-gray-700"
                    title={`Agent ${agentId}`}
                  >
                    <span className="h-1.5 w-1.5 rounded-full bg-blue-400" />
                    {name || `#${agentId.slice(0, 6)}`}
                  </span>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function formatEventType(eventType: string): string {
  return eventType
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

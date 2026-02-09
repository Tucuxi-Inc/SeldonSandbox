import { useSimulationStore } from '../../stores/simulation';

export function Header() {
  const { sessions, activeSessionId, setActiveSession } = useSimulationStore();
  const active = sessions.find((s) => s.id === activeSessionId);

  return (
    <header className="flex h-12 items-center justify-between border-b border-gray-800 bg-gray-950 px-4">
      <div className="flex items-center gap-4">
        <select
          className="rounded-md border border-gray-700 bg-gray-900 px-3 py-1 text-sm text-gray-200 focus:border-blue-500 focus:outline-none"
          value={activeSessionId || ''}
          onChange={(e) => setActiveSession(e.target.value || null)}
        >
          <option value="">Select session...</option>
          {sessions.map((s) => (
            <option key={s.id} value={s.id}>
              {s.name} ({s.id})
            </option>
          ))}
        </select>
      </div>

      {active && (
        <div className="flex items-center gap-4 text-sm">
          <span className="text-gray-400">
            Gen <span className="font-mono text-gray-200">{active.current_generation}</span>
            <span className="text-gray-600"> / {active.max_generations}</span>
          </span>
          <span className="text-gray-400">
            Pop <span className="font-mono text-gray-200">{active.population_size}</span>
          </span>
          <StatusBadge status={active.status} />
        </div>
      )}
    </header>
  );
}

function StatusBadge({ status }: { status: string }) {
  const colors: Record<string, string> = {
    created: 'bg-gray-700 text-gray-300',
    running: 'bg-blue-900 text-blue-300',
    completed: 'bg-green-900 text-green-300',
  };
  return (
    <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${colors[status] || colors.created}`}>
      {status}
    </span>
  );
}

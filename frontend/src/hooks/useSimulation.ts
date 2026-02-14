import { useEffect, useRef } from 'react';
import { useSimulationStore } from '../stores/simulation';

export function usePolling(intervalMs: number = 1000) {
  const { activeSessionId, sessions, refreshAll } = useSimulationStore();
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const activeSession = sessions.find((s) => s.id === activeSessionId);
  const isRunning = activeSession?.status === 'running';

  useEffect(() => {
    if (isRunning) {
      timerRef.current = setInterval(refreshAll, intervalMs);
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [isRunning, intervalMs, refreshAll]);
}

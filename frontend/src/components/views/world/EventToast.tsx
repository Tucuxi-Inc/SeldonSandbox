import { useEffect } from 'react';
import { X } from 'lucide-react';
import type { WorldEvent } from '../../../stores/playback';

interface EventToastProps {
  events: WorldEvent[];
  onDismiss: (id: string) => void;
}

const EVENT_STYLES: Record<string, { bg: string; border: string; text: string }> = {
  birth: { bg: 'bg-green-900/80', border: 'border-green-700', text: 'text-green-200' },
  death: { bg: 'bg-red-900/80', border: 'border-red-700', text: 'text-red-200' },
  pairing: { bg: 'bg-pink-900/80', border: 'border-pink-700', text: 'text-pink-200' },
};

export function EventToast({ events, onDismiss }: EventToastProps) {
  // Auto-dismiss after 5 seconds
  useEffect(() => {
    if (events.length === 0) return;

    const timer = setInterval(() => {
      const now = Date.now();
      for (const evt of events) {
        if (now - evt.timestamp > 5000) {
          onDismiss(evt.id);
        }
      }
    }, 1000);

    return () => clearInterval(timer);
  }, [events, onDismiss]);

  if (events.length === 0) return null;

  return (
    <div className="absolute right-3 top-3 z-20 flex flex-col gap-1.5 pointer-events-auto" style={{ maxWidth: 260 }}>
      {events.slice(0, 5).map(evt => {
        const style = EVENT_STYLES[evt.type] || EVENT_STYLES.birth;
        return (
          <div
            key={evt.id}
            className={`flex items-center gap-2 rounded-lg border px-3 py-1.5 text-xs shadow-lg ${style.bg} ${style.border} ${style.text}`}
          >
            <span className="flex-1">{evt.message}</span>
            <button onClick={() => onDismiss(evt.id)} className="opacity-60 hover:opacity-100">
              <X size={12} />
            </button>
          </div>
        );
      })}
    </div>
  );
}

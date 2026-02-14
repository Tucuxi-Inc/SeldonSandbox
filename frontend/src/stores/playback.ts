import { create } from 'zustand';
import type { TickStateResponse, AgentTickActivity } from '../types';
import * as api from '../api/client';

export interface WorldEvent {
  id: string;
  type: string;
  message: string;
  timestamp: number;
}

interface PlaybackStore {
  // State
  isPlaying: boolean;
  speed: number; // ms between ticks
  currentTick: TickStateResponse | null;
  previousTick: TickStateResponse | null;
  selectedAgentId: string | null;
  events: WorldEvent[];
  tickHistory: Map<string, { locations: (number[] | null)[] }>;
  loading: boolean;

  // Actions
  play: (sessionId: string) => void;
  pause: () => void;
  stepForward: (sessionId: string) => Promise<void>;
  setSpeed: (ms: number) => void;
  selectAgent: (id: string | null) => void;
  dismissEvent: (id: string) => void;
  loadTickState: (sessionId: string) => Promise<void>;
  reset: () => void;
}

let playInterval: ReturnType<typeof setInterval> | null = null;

export const usePlaybackStore = create<PlaybackStore>((set, get) => ({
  isPlaying: false,
  speed: 1000,
  currentTick: null,
  previousTick: null,
  selectedAgentId: null,
  events: [],
  tickHistory: new Map(),
  loading: false,

  play: (sessionId) => {
    const { isPlaying, speed } = get();
    if (isPlaying) return;

    set({ isPlaying: true });
    playInterval = setInterval(() => {
      get().stepForward(sessionId);
    }, speed);
  },

  pause: () => {
    if (playInterval) {
      clearInterval(playInterval);
      playInterval = null;
    }
    set({ isPlaying: false });
  },

  stepForward: async (sessionId) => {
    const { loading, currentTick } = get();
    if (loading) return;

    set({ loading: true });
    try {
      const data = await api.stepTick(sessionId);

      // Build events from tick data
      const newEvents: WorldEvent[] = [];
      for (const evt of data.events || []) {
        const id = `${data.global_tick}-${evt.type}-${(evt as Record<string, unknown>).agent_id || (evt as Record<string, unknown>).child_id || Math.random()}`;
        if (evt.type === 'birth') {
          const e = evt as Record<string, unknown>;
          newEvents.push({
            id,
            type: 'birth',
            message: `${e.child_name} was born`,
            timestamp: Date.now(),
          });
        } else if (evt.type === 'death') {
          const e = evt as Record<string, unknown>;
          newEvents.push({
            id,
            type: 'death',
            message: `${e.agent_name} has died (age ${e.age})`,
            timestamp: Date.now(),
          });
        } else if (evt.type === 'pairing') {
          newEvents.push({
            id,
            type: 'pairing',
            message: 'A new pair has formed',
            timestamp: Date.now(),
          });
        }
      }

      // Update tick history for trail rendering
      const history = new Map(get().tickHistory);
      if (data.agent_activities) {
        for (const [aid, ata] of Object.entries(data.agent_activities)) {
          const entry = history.get(aid) || { locations: [] };
          entry.locations.push((ata as AgentTickActivity).location);
          if (entry.locations.length > 12) entry.locations.shift();
          history.set(aid, entry);
        }
      }

      set({
        previousTick: currentTick,
        currentTick: data,
        events: [...newEvents, ...get().events].slice(0, 20),
        tickHistory: history,
        loading: false,
      });
    } catch {
      set({ loading: false });
    }
  },

  setSpeed: (ms) => {
    const { isPlaying } = get();
    set({ speed: ms });
    // Restart interval with new speed if playing
    if (isPlaying && playInterval) {
      clearInterval(playInterval);
      // Need to get sessionId from somewhere - use a workaround:
      // The play() function will be called again from the component
    }
  },

  selectAgent: (id) => set({ selectedAgentId: id }),

  dismissEvent: (id) => {
    set({ events: get().events.filter(e => e.id !== id) });
  },

  loadTickState: async (sessionId) => {
    try {
      const data = await api.getTickState(sessionId);
      if (data.enabled) {
        set({ currentTick: data });
      }
    } catch {
      // ignore
    }
  },

  reset: () => {
    if (playInterval) {
      clearInterval(playInterval);
      playInterval = null;
    }
    set({
      isPlaying: false,
      currentTick: null,
      previousTick: null,
      selectedAgentId: null,
      events: [],
      tickHistory: new Map(),
      loading: false,
    });
  },
}));

import { create } from 'zustand';
import type { SessionSummary, GenerationMetrics, AgentSummary } from '../types';
import * as api from '../api/client';

interface SimulationStore {
  // Sessions
  sessions: SessionSummary[];
  activeSessionId: string | null;

  // Data for active session
  metrics: GenerationMetrics[];
  agents: AgentSummary[];
  selectedAgentId: string | null;

  // Status
  loading: boolean;

  // Actions
  setActiveSession: (id: string | null) => void;
  selectAgent: (id: string | null) => void;
  refreshSessions: () => Promise<void>;
  refreshMetrics: () => Promise<void>;
  refreshAgents: () => Promise<void>;
  refreshAll: () => Promise<void>;
  updateSession: (summary: SessionSummary) => void;
}

export const useSimulationStore = create<SimulationStore>((set, get) => ({
  sessions: [],
  activeSessionId: null,
  metrics: [],
  agents: [],
  selectedAgentId: null,
  loading: false,

  setActiveSession: (id) => {
    set({ activeSessionId: id, metrics: [], agents: [], selectedAgentId: null });
    if (id) {
      get().refreshAll();
    }
  },

  selectAgent: (id) => set({ selectedAgentId: id }),

  refreshSessions: async () => {
    const sessions = await api.listSessions();
    set({ sessions });
  },

  refreshMetrics: async () => {
    const { activeSessionId } = get();
    if (!activeSessionId) return;
    try {
      const metrics = await api.getGenerations(activeSessionId);
      set({ metrics });
    } catch {
      // Session may not have metrics yet
    }
  },

  refreshAgents: async () => {
    const { activeSessionId } = get();
    if (!activeSessionId) return;
    try {
      const result = await api.listAgents(activeSessionId, { page_size: 200 });
      set({ agents: result.agents });
    } catch {
      // Session may not have agents yet
    }
  },

  refreshAll: async () => {
    set({ loading: true });
    await Promise.all([get().refreshSessions(), get().refreshMetrics(), get().refreshAgents()]);
    set({ loading: false });
  },

  updateSession: (summary) => {
    const sessions = get().sessions.map((s) =>
      s.id === summary.id ? summary : s,
    );
    set({ sessions });
  },
}));

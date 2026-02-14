import { useState } from 'react';
import { X, MessageCircle } from 'lucide-react';
import type { AgentTickActivity } from '../../../types';
import { REGION_COLORS, ACTIVITY_ICONS, ACTIVITY_LABELS } from '../../../lib/constants';
import * as api from '../../../api/client';

interface AgentDetailPanelProps {
  sessionId: string;
  agentId: string;
  agentName: string;
  activity: AgentTickActivity;
  season: string;
  onClose: () => void;
}

function NeedBar({ name, value }: { name: string; value: number }) {
  const pct = Math.max(0, Math.min(100, value * 100));
  const color = pct > 60 ? '#10B981' : pct > 30 ? '#F59E0B' : '#EF4444';
  return (
    <div className="flex items-center gap-2">
      <span className="w-14 text-[10px] text-gray-500 capitalize">{name}</span>
      <div className="flex-1 h-2 rounded-full bg-gray-800 overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${pct}%`, backgroundColor: color }}
        />
      </div>
      <span className="w-8 text-right text-[10px] text-gray-500">{pct.toFixed(0)}%</span>
    </div>
  );
}

export function AgentDetailPanel({
  sessionId,
  agentId,
  agentName,
  activity,
  season,
  onClose,
}: AgentDetailPanelProps) {
  const regionColor = REGION_COLORS[activity.processing_region] || '#6B7280';
  const [thought, setThought] = useState<string | null>(null);
  const [thoughtLoading, setThoughtLoading] = useState(false);

  const askThought = async () => {
    setThoughtLoading(true);
    setThought(null);
    try {
      // Build a contextual question based on current state
      const needsList = Object.entries(activity.needs_snapshot || {})
        .map(([n, v]) => `${n}: ${((v as number) * 100).toFixed(0)}%`)
        .join(', ');
      const question = `It is ${season}. You are currently ${
        activity.activity ? ACTIVITY_LABELS[activity.activity] || activity.activity : 'idle'
      }. Your needs: ${needsList}. Your health is ${(activity.health * 100).toFixed(0)}%. What are you thinking about right now? Reply in 1-2 sentences, in character.`;

      const resp = await api.interviewAgent(sessionId, agentId, question);
      setThought(resp.response);
    } catch (err: unknown) {
      setThought('(Could not reach LLM provider)');
    }
    setThoughtLoading(false);
  };

  return (
    <div className="w-72 overflow-y-auto rounded-lg border border-gray-800 bg-gray-900 p-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div>
          <h3 className="text-sm font-semibold text-gray-200">{agentName}</h3>
          <div className="flex items-center gap-2 mt-0.5">
            <span
              className="rounded px-1.5 py-0.5 text-[10px] font-medium"
              style={{
                backgroundColor: regionColor + '33',
                color: regionColor,
              }}
            >
              {activity.processing_region}
            </span>
            <span className="text-[10px] text-gray-500 capitalize">{activity.life_phase}</span>
            {activity.is_pregnant && (
              <span className="text-[10px] text-pink-400">Expecting</span>
            )}
          </div>
        </div>
        <button
          onClick={onClose}
          className="text-xs text-gray-500 hover:text-gray-300"
        >
          <X size={14} />
        </button>
      </div>

      {/* Current activity */}
      {activity.activity && (
        <div className="mb-3 rounded-md border border-gray-800 bg-gray-950 px-3 py-2">
          <div className="text-[10px] text-gray-500 mb-0.5">Current Activity</div>
          <div className="flex items-center gap-2 text-sm text-gray-200">
            <span>{ACTIVITY_ICONS[activity.activity] || ''}</span>
            <span>{ACTIVITY_LABELS[activity.activity] || activity.activity}</span>
          </div>
          {activity.activity_need && (
            <div className="text-[10px] text-gray-500 mt-0.5">
              Need: <span className="capitalize">{activity.activity_need}</span>
            </div>
          )}
        </div>
      )}

      {/* Health & Suffering */}
      <div className="mb-3 grid grid-cols-2 gap-2">
        <div className="rounded-md border border-gray-800 bg-gray-950 px-2 py-1.5 text-center">
          <div className="text-[10px] text-gray-500">Health</div>
          <div className={`text-sm font-semibold ${activity.health > 0.6 ? 'text-green-400' : activity.health > 0.3 ? 'text-yellow-400' : 'text-red-400'}`}>
            {(activity.health * 100).toFixed(0)}%
          </div>
        </div>
        <div className="rounded-md border border-gray-800 bg-gray-950 px-2 py-1.5 text-center">
          <div className="text-[10px] text-gray-500">Suffering</div>
          <div className={`text-sm font-semibold ${activity.suffering < 0.3 ? 'text-gray-400' : activity.suffering < 0.6 ? 'text-yellow-400' : 'text-red-400'}`}>
            {(activity.suffering * 100).toFixed(0)}%
          </div>
        </div>
      </div>

      {/* Location */}
      {activity.location && (
        <div className="mb-3 text-xs text-gray-500">
          Location: ({activity.location[0]}, {activity.location[1]})
          {activity.previous_location && (
            <span className="ml-1 text-gray-600">
              from ({activity.previous_location[0]}, {activity.previous_location[1]})
            </span>
          )}
        </div>
      )}

      {/* Needs */}
      <div className="space-y-1.5">
        <div className="text-[10px] font-semibold text-gray-400">Needs</div>
        {Object.entries(activity.needs_snapshot || {}).map(([name, value]) => (
          <NeedBar key={name} name={name} value={value as number} />
        ))}
      </div>

      {/* LLM Thought Bubble */}
      <div className="mt-3">
        <button
          onClick={askThought}
          disabled={thoughtLoading}
          className="flex w-full items-center justify-center gap-1.5 rounded-md border border-gray-700 bg-gray-800 px-3 py-1.5 text-xs text-gray-300 hover:bg-gray-700 transition-colors disabled:opacity-50"
        >
          <MessageCircle size={12} />
          {thoughtLoading ? 'Thinking...' : 'What are they thinking?'}
        </button>
        {thought && (
          <div className="mt-2 rounded-md border border-blue-800/40 bg-blue-950/30 px-3 py-2 text-xs text-blue-200 italic">
            "{thought}"
          </div>
        )}
      </div>

      {/* Agent ID for reference */}
      <div className="mt-3 text-[10px] text-gray-600 font-mono">
        {agentId}
      </div>
    </div>
  );
}

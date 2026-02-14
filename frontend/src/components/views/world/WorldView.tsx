import { useState, useEffect, useCallback } from 'react';
import { useSimulationStore } from '../../../stores/simulation';
import { usePlaybackStore } from '../../../stores/playback';
import { EmptyState } from '../../shared/EmptyState';
import { WorldToolbar } from './WorldToolbar';
import { PlaybackControls } from './PlaybackControls';
import { WorldHexGrid } from './WorldHexGrid';
import { AgentDetailPanel } from './AgentDetailPanel';
import { EventToast } from './EventToast';
import * as api from '../../../api/client';
import type { HexGridResponse } from '../../../types';

export function WorldView() {
  const { activeSessionId } = useSimulationStore();
  const {
    isPlaying,
    speed,
    currentTick,
    selectedAgentId,
    events,
    loading,
    tickHistory,
    play,
    pause,
    stepForward,
    setSpeed,
    selectAgent,
    dismissEvent,
    loadTickState,
    reset,
  } = usePlaybackStore();

  const [gridData, setGridData] = useState<HexGridResponse | null>(null);
  const [gridLoading, setGridLoading] = useState(false);
  const [colorMode, setColorMode] = useState<'terrain' | 'density' | 'region'>('terrain');
  const [showConnections, setShowConnections] = useState(true);

  // Load grid data and initial tick state
  useEffect(() => {
    if (!activeSessionId) {
      reset();
      setGridData(null);
      return;
    }

    setGridLoading(true);
    Promise.all([
      api.getHexGrid(activeSessionId),
      loadTickState(activeSessionId),
    ]).then(([grid]) => {
      setGridData(grid);
      setGridLoading(false);
    }).catch(() => {
      setGridLoading(false);
    });

    return () => {
      pause();
    };
  }, [activeSessionId]);

  // Handle speed changes while playing
  useEffect(() => {
    if (isPlaying && activeSessionId) {
      pause();
      play(activeSessionId);
    }
  }, [speed]);

  const handlePlay = useCallback(() => {
    if (activeSessionId) play(activeSessionId);
  }, [activeSessionId, play]);

  const handleStep = useCallback(() => {
    if (activeSessionId) stepForward(activeSessionId);
  }, [activeSessionId, stepForward]);

  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      // Don't capture when typing in inputs
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;

      switch (e.key) {
        case ' ':
          e.preventDefault();
          if (isPlaying) pause();
          else if (activeSessionId) play(activeSessionId);
          break;
        case 'ArrowRight':
          e.preventDefault();
          if (!isPlaying && activeSessionId) stepForward(activeSessionId);
          break;
        case '1':
          setSpeed(2000);
          break;
        case '2':
          setSpeed(1000);
          break;
        case '3':
          setSpeed(500);
          break;
        case '4':
          setSpeed(200);
          break;
        case 'Escape':
          selectAgent(null);
          break;
      }
    };

    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [isPlaying, activeSessionId, play, pause, stepForward, setSpeed, selectAgent]);

  if (!activeSessionId) {
    return <EmptyState message="Create or select a simulation session to use World View" />;
  }

  if (gridLoading) {
    return (
      <div className="flex h-full items-center justify-center text-gray-500">
        Loading world...
      </div>
    );
  }

  if (gridData && !gridData.enabled) {
    return (
      <div className="space-y-4">
        <h1 className="text-2xl font-bold text-gray-100">World</h1>
        <div className="rounded-lg border border-gray-800 bg-gray-900 p-8 text-center text-gray-500">
          Enable <code className="text-gray-400">tick_config.enabled</code> and{' '}
          <code className="text-gray-400">hex_grid_config.enabled</code> in session config to use World View.
        </div>
      </div>
    );
  }

  const tick = currentTick;
  const selectedActivity = selectedAgentId && tick?.agent_activities?.[selectedAgentId];
  const selectedName = selectedAgentId && tick?.agent_names?.[selectedAgentId];

  return (
    <div className="flex h-full flex-col gap-3">
      <WorldToolbar
        year={tick?.year ?? 0}
        season={tick?.season ?? 'spring'}
        tickInYear={tick?.tick_in_year ?? 0}
        populationCount={tick?.population_count ?? 0}
        sessionStatus={tick?.session_status ?? ''}
        colorMode={colorMode}
        onColorModeChange={setColorMode}
        showConnections={showConnections}
        onToggleConnections={() => setShowConnections(v => !v)}
      />

      <PlaybackControls
        isPlaying={isPlaying}
        speed={speed}
        globalTick={tick?.global_tick ?? 0}
        loading={loading}
        onPlay={handlePlay}
        onPause={pause}
        onStep={handleStep}
        onSpeedChange={setSpeed}
      />

      <div className="flex flex-1 gap-4 min-h-0">
        {gridData && (
          <div className="relative flex-1">
            <WorldHexGrid
              gridData={gridData}
              tickState={tick ?? null}
              colorMode={colorMode}
              selectedAgentId={selectedAgentId}
              showConnections={showConnections}
              tickHistory={tickHistory}
              onSelectAgent={selectAgent}
            />
            <EventToast events={events} onDismiss={dismissEvent} />
          </div>
        )}

        {selectedActivity && selectedName && (
          <AgentDetailPanel
            sessionId={activeSessionId}
            agentId={selectedAgentId!}
            agentName={selectedName}
            activity={selectedActivity}
            season={tick?.season ?? 'spring'}
            onClose={() => selectAgent(null)}
          />
        )}
      </div>
    </div>
  );
}

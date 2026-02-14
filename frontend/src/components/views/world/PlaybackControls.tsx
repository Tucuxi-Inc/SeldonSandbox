import { Play, Pause, SkipForward } from 'lucide-react';
import { SPEED_PRESETS } from '../../../lib/constants';

interface PlaybackControlsProps {
  isPlaying: boolean;
  speed: number;
  globalTick: number;
  loading: boolean;
  onPlay: () => void;
  onPause: () => void;
  onStep: () => void;
  onSpeedChange: (ms: number) => void;
}

export function PlaybackControls({
  isPlaying,
  speed,
  globalTick,
  loading,
  onPlay,
  onPause,
  onStep,
  onSpeedChange,
}: PlaybackControlsProps) {
  return (
    <div className="flex items-center justify-between rounded-lg border border-gray-800 bg-gray-900 px-4 py-2">
      <div className="flex items-center gap-2">
        <button
          onClick={isPlaying ? onPause : onPlay}
          className="flex h-8 w-8 items-center justify-center rounded-md bg-blue-600 text-white hover:bg-blue-500 transition-colors"
          title={isPlaying ? 'Pause' : 'Play'}
        >
          {isPlaying ? <Pause size={16} /> : <Play size={16} />}
        </button>
        <button
          onClick={onStep}
          disabled={isPlaying || loading}
          className="flex h-8 w-8 items-center justify-center rounded-md bg-gray-800 text-gray-300 hover:bg-gray-700 transition-colors disabled:opacity-40"
          title="Step forward one tick"
        >
          <SkipForward size={16} />
        </button>
      </div>

      <div className="flex items-center gap-2">
        <span className="text-xs text-gray-500">Speed:</span>
        {SPEED_PRESETS.map(preset => (
          <button
            key={preset.label}
            onClick={() => onSpeedChange(preset.ms)}
            className={`rounded px-2 py-1 text-xs font-medium transition-colors ${
              speed === preset.ms
                ? 'bg-blue-600/20 text-blue-400'
                : 'text-gray-500 hover:text-gray-300'
            }`}
          >
            {preset.label}
          </button>
        ))}
      </div>

      <div className="text-xs text-gray-500">
        Tick #{globalTick}
        {loading && <span className="ml-2 text-blue-400">...</span>}
      </div>
    </div>
  );
}

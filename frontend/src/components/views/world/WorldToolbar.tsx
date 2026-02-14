import { SEASON_COLORS } from '../../../lib/constants';

interface WorldToolbarProps {
  year: number;
  season: string;
  tickInYear: number;
  populationCount: number;
  colorMode: 'terrain' | 'density' | 'region';
  onColorModeChange: (mode: 'terrain' | 'density' | 'region') => void;
  showConnections: boolean;
  onToggleConnections: () => void;
}

const SEASON_LABELS: Record<string, string> = {
  spring: 'Spring',
  summer: 'Summer',
  autumn: 'Autumn',
  winter: 'Winter',
};

export function WorldToolbar({
  year,
  season,
  tickInYear,
  populationCount,
  colorMode,
  onColorModeChange,
  showConnections,
  onToggleConnections,
}: WorldToolbarProps) {
  return (
    <div className="flex items-center justify-between">
      <div className="flex items-center gap-4">
        <h1 className="text-2xl font-bold text-gray-100">World</h1>
        <div className="flex items-center gap-3 text-sm text-gray-400">
          <span>Year <span className="font-semibold text-gray-200">{year}</span></span>
          <span
            className="rounded px-2 py-0.5 text-xs font-medium"
            style={{ backgroundColor: SEASON_COLORS[season] || 'transparent', color: '#e5e7eb' }}
          >
            {SEASON_LABELS[season] || season}
          </span>
          <span>Tick {tickInYear + 1}/12</span>
          <span>Pop: <span className="font-semibold text-gray-200">{populationCount}</span></span>
        </div>
      </div>
      <div className="flex items-center gap-2">
        {(['terrain', 'density', 'region'] as const).map(mode => (
          <button
            key={mode}
            onClick={() => onColorModeChange(mode)}
            className={`rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
              colorMode === mode
                ? 'bg-blue-600 text-white'
                : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
            }`}
          >
            {mode.charAt(0).toUpperCase() + mode.slice(1)}
          </button>
        ))}
        <div className="mx-1 h-4 border-l border-gray-700" />
        <button
          onClick={onToggleConnections}
          className={`rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
            showConnections
              ? 'bg-red-600/20 text-red-400'
              : 'bg-gray-800 text-gray-500 hover:bg-gray-700'
          }`}
        >
          Links
        </button>
      </div>
    </div>
  );
}

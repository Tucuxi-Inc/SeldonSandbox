import { REGION_COLORS, REGION_LABELS } from '../../lib/constants';

export function RegionBadge({ region }: { region: string }) {
  const color = REGION_COLORS[region] || '#6B7280';
  const label = REGION_LABELS[region] || region;
  return (
    <span
      className="inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium"
      style={{ backgroundColor: color + '22', color, border: `1px solid ${color}44` }}
    >
      {label}
    </span>
  );
}

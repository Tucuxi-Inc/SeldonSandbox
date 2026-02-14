export const REGION_COLORS: Record<string, string> = {
  under_processing: '#6B7280',
  optimal: '#10B981',
  deep: '#3B82F6',
  sacrificial: '#F59E0B',
  pathological: '#EF4444',
};

export const REGION_LABELS: Record<string, string> = {
  under_processing: 'R1: Under-Processing',
  optimal: 'R2: Optimal',
  deep: 'R3: Deep',
  sacrificial: 'R4: Sacrificial',
  pathological: 'R5: Pathological',
};

export const REGION_ORDER = [
  'under_processing',
  'optimal',
  'deep',
  'sacrificial',
  'pathological',
];

export const API_BASE = '/api';

export const ANOMALY_COLORS: Record<string, string> = {
  normal: '#10B981',
  warning: '#F59E0B',
  medium: '#F97316',
  high: '#EF4444',
  critical: '#DC2626',
};

export const MEMORY_TYPE_COLORS: Record<string, string> = {
  personal: '#3B82F6',
  family: '#8B5CF6',
  societal: '#F59E0B',
  myth: '#EF4444',
};

export const EDGE_TYPE_COLORS: Record<string, string> = {
  partner: '#EF4444',
  social: '#3B82F6',
  parent: '#10B981',
};

export const SENSITIVITY_COLORS = {
  positive: '#3B82F6',
  negative: '#EF4444',
};

export const EPISTEMOLOGY_COLORS: Record<string, string> = {
  empirical: '#3B82F6',
  traditional: '#F59E0B',
  sacred: '#8B5CF6',
  mythical: '#EF4444',
};

export const BELIEF_DOMAIN_COLORS: Record<string, string> = {
  resource: '#10B981',
  danger: '#EF4444',
  social: '#3B82F6',
  productivity: '#F59E0B',
  migration: '#8B5CF6',
  reproduction: '#EC4899',
};

export const EXPERIENCE_DIM_COLORS: Record<string, string> = {
  valence: '#10B981',
  arousal: '#EF4444',
  social_quality: '#3B82F6',
  agency: '#F59E0B',
  novelty: '#8B5CF6',
  meaning: '#EC4899',
};

export const EXPERIENCE_DIM_LABELS: Record<string, string> = {
  valence: 'Valence',
  arousal: 'Arousal',
  social_quality: 'Social Quality',
  agency: 'Agency',
  novelty: 'Novelty',
  meaning: 'Meaning',
};

export const PQ_BUCKET_COLORS: Record<string, string> = {
  '0.0-0.2': '#EF4444',
  '0.2-0.4': '#F97316',
  '0.4-0.6': '#F59E0B',
  '0.6-0.8': '#22C55E',
  '0.8-1.0': '#10B981',
};

// === Hex Grid Terrain ===

export const TERRAIN_COLORS: Record<string, string> = {
  ocean: '#1e3a5f',
  coast: '#4a90c4',
  coastal_valley: '#7ab648',
  foothills: '#8b7355',
  mountains: '#6b6b6b',
  high_desert: '#c4a35a',
  desert: '#d4a853',
  forest: '#2d5a27',
  plains: '#a8c256',
  river_valley: '#3d8b37',
};

export const TERRAIN_LABELS: Record<string, string> = {
  ocean: 'Ocean',
  coast: 'Coast',
  coastal_valley: 'Coastal Valley',
  foothills: 'Foothills',
  mountains: 'Mountains',
  high_desert: 'High Desert',
  desert: 'Desert',
  forest: 'Forest',
  plains: 'Plains',
  river_valley: 'River Valley',
};

// === Chronicle / Biography ===

export const SEVERITY_COLORS: Record<string, string> = {
  minor: '#6B7280',
  notable: '#3B82F6',
  major: '#F59E0B',
  critical: '#EF4444',
};

export const COMPARISON_COLORS = ['#3B82F6', '#F59E0B', '#10B981'];

// === World View ===

export const ACTIVITY_ICONS: Record<string, string> = {
  forage: '\u{1F33F}',
  hunt: '\u{1F3F9}',
  fish: '\u{1F41F}',
  find_water: '\u{1F4A7}',
  build_shelter: '\u{1F3E0}',
  seek_warmth: '\u{1F525}',
  rest: '\u{1F4A4}',
  seek_safety: '\u{1F6E1}',
};

export const ACTIVITY_LABELS: Record<string, string> = {
  forage: 'Foraging',
  hunt: 'Hunting',
  fish: 'Fishing',
  find_water: 'Finding Water',
  build_shelter: 'Building Shelter',
  seek_warmth: 'Seeking Warmth',
  rest: 'Resting',
  seek_safety: 'Seeking Safety',
};

export const SEASON_COLORS: Record<string, string> = {
  spring: 'rgba(34, 197, 94, 0.05)',
  summer: 'rgba(250, 204, 21, 0.05)',
  autumn: 'rgba(249, 115, 22, 0.05)',
  winter: 'rgba(96, 165, 250, 0.08)',
};

export const LIFE_PHASE_SIZES: Record<string, number> = {
  infant: 2,
  child: 3,
  adolescent: 3.5,
  young_adult: 4,
  mature: 5,
  middle_aged: 4.5,
  elder: 4,
  ancient: 3.5,
};

export const SPEED_PRESETS = [
  { label: '1x', ms: 2000 },
  { label: '2x', ms: 1000 },
  { label: '4x', ms: 500 },
  { label: '10x', ms: 200 },
];

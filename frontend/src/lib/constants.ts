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

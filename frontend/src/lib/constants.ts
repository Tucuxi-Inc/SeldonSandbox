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

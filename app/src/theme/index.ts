export const colors = {
  // Backgrounds
  bg: '#080808',
  bgCard: '#111111',
  bgElevated: '#1A1A1A',
  bgInput: '#161616',

  // Accent — electric violet
  accent: '#8B5CF6',
  accentMid: '#A78BFA',
  accentSoft: 'rgba(139, 92, 246, 0.15)',
  accentGlow: 'rgba(139, 92, 246, 0.35)',

  // Secondary accent — teal for "vibe" tags
  teal: '#2DD4BF',
  tealSoft: 'rgba(45, 212, 191, 0.15)',

  // Text
  textPrimary: '#F8F8F8',
  textSecondary: '#9CA3AF',
  textMuted: '#4B5563',

  // States
  success: '#34D399',
  error: '#F87171',
  warning: '#FBBF24',

  // Borders
  border: 'rgba(255,255,255,0.07)',
  borderFocused: 'rgba(139, 92, 246, 0.5)',
} as const;

export const spacing = {
  xxs: 2,
  xs: 4,
  sm: 8,
  md: 16,
  lg: 24,
  xl: 32,
  xxl: 48,
} as const;

export const radius = {
  sm: 8,
  md: 12,
  lg: 16,
  xl: 24,
  full: 9999,
} as const;

export const typography = {
  displayLg: { fontSize: 36, fontWeight: '700' as const, letterSpacing: -1 },
  displayMd: { fontSize: 28, fontWeight: '700' as const, letterSpacing: -0.5 },
  headingLg: { fontSize: 22, fontWeight: '600' as const, letterSpacing: -0.3 },
  headingMd: { fontSize: 18, fontWeight: '600' as const },
  bodyLg: { fontSize: 16, fontWeight: '400' as const },
  bodyMd: { fontSize: 14, fontWeight: '400' as const },
  bodySm: { fontSize: 12, fontWeight: '400' as const },
  label: { fontSize: 11, fontWeight: '600' as const, letterSpacing: 0.8 },
} as const;

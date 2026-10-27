import React from 'react';
import { Text, StyleSheet, View } from 'react-native';
import { colors, radius, spacing, typography } from '../theme';

const VIBE_COLORS: Record<string, { bg: string; text: string }> = {
  slow: { bg: 'rgba(139,92,246,0.15)', text: '#A78BFA' },
  heavy: { bg: 'rgba(139,92,246,0.15)', text: '#A78BFA' },
  sparse: { bg: 'rgba(45,212,191,0.15)', text: '#2DD4BF' },
  intimate: { bg: 'rgba(45,212,191,0.15)', text: '#2DD4BF' },
  hushed: { bg: 'rgba(148,163,184,0.15)', text: '#CBD5E1' },
  laid: { bg: 'rgba(251,191,36,0.15)', text: '#FCD34D' },
  raw: { bg: 'rgba(251,146,60,0.15)', text: '#FB923C' },
  psychedelic: { bg: 'rgba(232,121,249,0.15)', text: '#E879F9' },
  punk: { bg: 'rgba(239,68,68,0.15)', text: '#FCA5A5' },
  lean: { bg: 'rgba(196,181,253,0.15)', text: '#C4B5FD' },
  dark: { bg: 'rgba(99,102,241,0.15)', text: '#A5B4FC' },
  bright: { bg: 'rgba(251,191,36,0.15)', text: '#FCD34D' },
  folk: { bg: 'rgba(217,119,6,0.15)', text: '#FBBF24' },
  dance: { bg: 'rgba(168,85,247,0.15)', text: '#D8B4FE' },
  reggaeton: { bg: 'rgba(234,88,12,0.15)', text: '#FB923C' },
  reggae: { bg: 'rgba(34,197,94,0.15)', text: '#4ADE80' },
  warm: { bg: 'rgba(251,146,60,0.15)', text: '#FDBA74' },
  driving: { bg: 'rgba(239,68,68,0.15)', text: '#FCA5A5' },
  chill: { bg: 'rgba(45,212,191,0.15)', text: '#2DD4BF' },
};

const DEFAULT_COLOR = { bg: 'rgba(255,255,255,0.08)', text: colors.textSecondary };

interface Props {
  label: string;
}

export const VibeTag: React.FC<Props> = ({ label }) => {
  const firstWord = label.trim().split(/[\s-]+/)[0].toLowerCase();
  let c = VIBE_COLORS[firstWord];
  if (!c) {
    const l = label.toLowerCase();
    for (const [key, val] of Object.entries(VIBE_COLORS)) {
      if (l.includes(key)) {
        c = val;
        break;
      }
    }
  }
  const color = c ?? DEFAULT_COLOR;
  return (
    <View style={[styles.tag, { backgroundColor: color.bg }]}>
      <Text style={[styles.text, { color: color.text }]}>{label}</Text>
    </View>
  );
};

const styles = StyleSheet.create({
  tag: {
    paddingHorizontal: spacing.sm,
    paddingVertical: spacing.xs,
    borderRadius: radius.full,
    borderWidth: 0.5,
  },
  text: {
    ...typography.label,
    textTransform: 'uppercase',
  },
});

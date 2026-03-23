import React from 'react';
import { Text, StyleSheet, View } from 'react-native';
import { colors, radius, typography } from '../theme';

const VIBE_COLORS: Record<string, { bg: string; border: string; text: string }> = {
  dark:       { bg: 'rgba(139,92,246,0.12)',  border: 'rgba(139,92,246,0.4)',  text: '#A78BFA' },
  energetic:  { bg: 'rgba(251,191,36,0.12)',  border: 'rgba(251,191,36,0.4)',  text: '#FCD34D' },
  chill:      { bg: 'rgba(45,212,191,0.12)',  border: 'rgba(45,212,191,0.4)',  text: '#2DD4BF' },
  melodic:    { bg: 'rgba(96,165,250,0.12)',  border: 'rgba(96,165,250,0.4)',  text: '#93C5FD' },
  aggressive: { bg: 'rgba(248,113,113,0.12)', border: 'rgba(248,113,113,0.4)', text: '#FCA5A5' },
  euphoric:   { bg: 'rgba(196,181,253,0.12)', border: 'rgba(196,181,253,0.4)', text: '#C4B5FD' },
  sad:        { bg: 'rgba(99,102,241,0.12)',  border: 'rgba(99,102,241,0.4)',  text: '#A5B4FC' },
  groovy:     { bg: 'rgba(52,211,153,0.12)',  border: 'rgba(52,211,153,0.4)',  text: '#34D399' },
  atmospheric:{ bg: 'rgba(148,163,184,0.12)', border: 'rgba(148,163,184,0.4)', text: '#CBD5E1' },
  hypnotic:   { bg: 'rgba(232,121,249,0.12)', border: 'rgba(232,121,249,0.4)', text: '#E879F9' },
};

const DEFAULT_COLOR = { bg: 'rgba(255,255,255,0.06)', border: 'rgba(255,255,255,0.15)', text: colors.textSecondary };

interface Props {
  label: string;
}

export const VibeTag: React.FC<Props> = ({ label }) => {
  const c = VIBE_COLORS[label.toLowerCase()] ?? DEFAULT_COLOR;
  return (
    <View style={[styles.tag, { backgroundColor: c.bg, borderColor: c.border }]}>
      <Text style={[styles.text, { color: c.text }]}>{label}</Text>
    </View>
  );
};

const styles = StyleSheet.create({
  tag: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: radius.full,
    borderWidth: 0.5,
  },
  text: {
    ...typography.label,
    textTransform: 'uppercase',
  },
});

import React from 'react';
import {
  View,
  Text,
  Image,
  StyleSheet,
  TouchableOpacity,
  Dimensions,
  ActivityIndicator,
  Platform,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { BlurView } from 'expo-blur';
import { colors, radius, spacing, typography } from '../theme';
import { VibeTag } from './VibeTag';
import type { TrackResult } from '../services/api';

const { width: SCREEN_W } = Dimensions.get('window');

interface Props {
  track: TrackResult;
  variant?: 'hero' | 'compact';
  onPress?: () => void;
  rank?: number;
  loading?: boolean;
}

export const SongCard: React.FC<Props> = ({ track, variant = 'compact', onPress, rank, loading }) => {
  if (variant === 'hero') {
    return (
      <TouchableOpacity activeOpacity={0.85} onPress={onPress} style={styles.heroWrapper}>
        <Image
          source={{ uri: track.albumArtUrl }}
          style={styles.heroImage}
          resizeMode="cover"
        />
        <LinearGradient
          colors={['transparent', 'rgba(8,8,8,0.55)', 'rgba(8,8,8,0.9)']}
          style={styles.heroGradient}
        />
        <View style={styles.heroContent}>
          {/* Glass panel behind hero content */}
          {Platform.OS === 'ios' ? (
            <BlurView intensity={12} tint="dark" style={[StyleSheet.absoluteFill, styles.heroGlassPanel]} />
          ) : (
            <View style={[StyleSheet.absoluteFill, styles.heroGlassPanelAndroid]} />
          )}
          <View style={[StyleSheet.absoluteFill, styles.heroGlassOverlay]} pointerEvents="none" />
          <View style={styles.heroGlassTopBorder} pointerEvents="none" />
          {track.matchScore != null && track.matchScore > 0 && (
            <View style={styles.matchBadge}>
              <Text style={styles.matchText}>{Math.round(track.matchScore * 100)}% match</Text>
            </View>
          )}
          <Text style={styles.heroTitle} numberOfLines={1}>{track.title}</Text>
          <Text style={styles.heroArtist} numberOfLines={1}>{track.artist}</Text>
          {track.vibes.length > 0 && (
            <View style={styles.tagRow}>
              {track.vibes.slice(0, 3).map((v) => (
                <VibeTag key={v} label={v} />
              ))}
            </View>
          )}
          {track.bpm > 0 && (
            <View style={styles.statsRow}>
              <Stat label="BPM" value={String(track.bpm)} />
              {track.key && track.key !== '—' && <Stat label="KEY" value={track.key} />}
              <Stat label="ENERGY" value={`${Math.round(track.energy * 100)}%`} />
              <Stat label="MOOD" value={track.valence > 0.5 ? 'Happy' : 'Dark'} />
            </View>
          )}
        </View>
      </TouchableOpacity>
    );
  }

  return (
    <TouchableOpacity activeOpacity={0.75} onPress={onPress} style={styles.card}>
      {/* Glass background */}
      {Platform.OS === 'ios' ? (
        <BlurView intensity={18} tint="dark" style={StyleSheet.absoluteFill} />
      ) : (
        <View style={[StyleSheet.absoluteFill, styles.androidCardBg]} />
      )}
      <View style={[StyleSheet.absoluteFill, styles.cardGlassOverlay]} pointerEvents="none" />
      <View style={styles.cardTopHighlight} pointerEvents="none" />

      <View style={styles.artContainer}>
        {rank !== undefined && (
          <View style={styles.rankBadge}>
            <Text style={styles.rankText}>{rank}</Text>
          </View>
        )}
        <Image
          source={{ uri: track.albumArtUrl }}
          style={styles.art}
          resizeMode="cover"
        />
      </View>
      <View style={styles.info}>
        <Text style={styles.title} numberOfLines={1}>{track.title}</Text>
        <Text style={styles.artist} numberOfLines={1}>{track.artist}</Text>
        <View style={styles.tagRow}>
          {track.vibes.slice(0, 2).map((v) => (
            <VibeTag key={v} label={v} />
          ))}
        </View>
      </View>
      <View style={styles.scoreContainer}>
        {loading ? (
          <ActivityIndicator size="small" color={colors.accent} />
        ) : track.matchScore != null && (
          <>
            <Text style={styles.scoreValue}>{Math.round(track.matchScore * 100)}</Text>
            <Text style={styles.scoreLabel}>%</Text>
          </>
        )}
      </View>
    </TouchableOpacity>
  );
};

const Stat: React.FC<{ label: string; value: string }> = ({ label, value }) => (
  <View style={styles.stat}>
    <Text style={styles.statValue}>{value}</Text>
    <Text style={styles.statLabel}>{label}</Text>
  </View>
);

const styles = StyleSheet.create({
  // Hero variant
  heroWrapper: {
    marginHorizontal: spacing.md,
    borderRadius: radius.xl,
    overflow: 'hidden',
    backgroundColor: colors.bgCard,
  },
  heroImage: {
    width: '100%',
    height: SCREEN_W - spacing.md * 2,
  },
  heroGradient: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    height: '70%',
  },
  heroContent: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    padding: spacing.lg,
    gap: spacing.sm,
    overflow: 'hidden',
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
  },
  heroGlassPanel: {
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
  },
  heroGlassPanelAndroid: {
    backgroundColor: 'rgba(8,8,8,0.75)',
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
  },
  heroGlassOverlay: {
    backgroundColor: 'rgba(255,255,255,0.03)',
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
  },
  heroGlassTopBorder: {
    position: 'absolute',
    top: 0,
    left: 24,
    right: 24,
    height: 1,
    backgroundColor: 'rgba(255,255,255,0.15)',
  },
  matchBadge: {
    alignSelf: 'flex-start',
    backgroundColor: colors.accentSoft,
    borderRadius: radius.full,
    paddingHorizontal: 10,
    paddingVertical: 3,
    borderWidth: 1,
    borderColor: colors.accentGlow,
  },
  matchText: {
    ...typography.label,
    color: colors.accentMid,
  },
  heroTitle: {
    ...typography.displayMd,
    color: colors.textPrimary,
  },
  heroArtist: {
    ...typography.bodyLg,
    color: colors.textSecondary,
    marginTop: -4,
  },
  tagRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 6,
  },
  statsRow: {
    flexDirection: 'row',
    gap: spacing.md,
    marginTop: spacing.xs,
  },
  stat: {
    alignItems: 'center',
  },
  statValue: {
    ...typography.bodyMd,
    color: colors.textPrimary,
    fontWeight: '600',
  },
  statLabel: {
    ...typography.label,
    color: colors.textMuted,
  },

  // Compact variant
  card: {
    flexDirection: 'row',
    alignItems: 'center',
    borderRadius: radius.lg,
    padding: spacing.md,
    gap: spacing.md,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.1)',
    overflow: 'hidden',
  },
  androidCardBg: {
    backgroundColor: 'rgba(18, 18, 22, 0.94)',
    borderRadius: radius.lg,
  },
  cardGlassOverlay: {
    backgroundColor: 'rgba(255, 255, 255, 0.04)',
    borderRadius: radius.lg,
  },
  cardTopHighlight: {
    position: 'absolute',
    top: 0,
    left: 12,
    right: 12,
    height: 1,
    backgroundColor: 'rgba(255, 255, 255, 0.14)',
  },
  artContainer: {
    position: 'relative',
  },
  rankBadge: {
    position: 'absolute',
    top: -6,
    left: -6,
    width: 20,
    height: 20,
    borderRadius: 10,
    backgroundColor: colors.accent,
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 1,
  },
  rankText: {
    ...typography.label,
    color: '#fff',
    fontSize: 10,
  },
  art: {
    width: 56,
    height: 56,
    borderRadius: radius.md,
    backgroundColor: colors.bgElevated,
  },
  info: {
    flex: 1,
    gap: 4,
  },
  title: {
    ...typography.bodyLg,
    color: colors.textPrimary,
    fontWeight: '600',
  },
  artist: {
    ...typography.bodyMd,
    color: colors.textSecondary,
  },
  scoreContainer: {
    alignItems: 'flex-end',
    flexDirection: 'row',
    gap: 1,
  },
  scoreValue: {
    ...typography.headingMd,
    color: colors.accent,
  },
  scoreLabel: {
    ...typography.bodyMd,
    color: colors.textMuted,
    marginBottom: 2,
  },
});

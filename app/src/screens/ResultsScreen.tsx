import React, { useCallback, useEffect, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  StatusBar,
  Alert,
  ActivityIndicator,
  Linking,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import type { RouteProp } from '@react-navigation/native';
import { useNavigation, useRoute } from '@react-navigation/native';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { LinearGradient } from 'expo-linear-gradient';

import { SongCard } from '../components/SongCard';
import { colors, spacing, typography } from '../theme';
import type { RootStackParamList } from '../navigation/AppNavigator';
import type { TrackResult } from '../services/api';
import { getRecommendations } from '../services/api';

type Nav = NativeStackNavigationProp<RootStackParamList>;
type Route = RouteProp<RootStackParamList, 'Results'>;

export const ResultsScreen: React.FC = () => {
  const navigation = useNavigation<Nav>();
  const { params } = useRoute<Route>();
  const { data, mode, query } = params;

  const identified = 'identified' in data ? data.identified : null;
  const searchResults = 'results' in data ? data.results : null;
  const initialRecs = data.recommendations;

  // search mode: recs are fetched on mount for the best match
  const [searchRecs, setSearchRecs] = useState<TrackResult[]>([]);
  const [recLoading, setRecLoading] = useState(mode === 'search');
  const [noPreview, setNoPreview] = useState(false);
  const [loadingTrackId, setLoadingTrackId] = useState<string | null>(null);

  // For identify/select, recs come pre-loaded; for search they're fetched below
  const displayRecs =
    mode === 'search' ? searchRecs : mode === 'select' ? initialRecs.slice(0, 5) : initialRecs; // identify: all recs

  const showNoPreview =
    (mode === 'search' && !recLoading && noPreview) ||
    (mode === 'select' && displayRecs.length === 0) ||
    (mode === 'identify' && !!identified && displayRecs.length === 0);

  useEffect(() => {
    if (mode !== 'search' || !searchResults?.[0]) {
      setRecLoading(false);
      return;
    }
    getRecommendations(searchResults[0].id)
      .then(fetched => {
        setSearchRecs(fetched.slice(0, 5));
        setNoPreview(fetched.length === 0);
      })
      .catch(() => setNoPreview(true))
      .finally(() => setRecLoading(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const openInDeezer = useCallback(async (track: TrackResult) => {
    const appUri = `deezer://www.deezer.com/track/${track.id}`;
    const webUrl = `https://www.deezer.com/track/${track.id}`;
    const canOpen = await Linking.canOpenURL(appUri);
    Linking.openURL(canOpen ? appUri : webUrl);
  }, []);

  const handleTrackSelect = useCallback(
    async (track: TrackResult) => {
      if (loadingTrackId) return;
      setLoadingTrackId(track.id);
      try {
        const recs = await getRecommendations(track.id);
        navigation.push('Results', {
          data: { identified: track, recommendations: recs },
          mode: 'select',
        });
      } catch (e: unknown) {
        Alert.alert('Could not load recommendations', (e as Error).message);
      } finally {
        setLoadingTrackId(null);
      }
    },
    [loadingTrackId, navigation],
  );

  const headerTitle =
    mode === 'select'
      ? 'Here you go'
      : mode === 'identify'
        ? identified
          ? 'Found it!'
          : 'No match found'
        : `Results for "${query}"`;

  return (
    <View style={styles.root}>
      <StatusBar barStyle="light-content" />

      <SafeAreaView style={styles.safe} edges={['top']}>
        {/* Nav bar */}
        <View style={styles.navbar}>
          <TouchableOpacity onPress={() => navigation.goBack()} style={styles.backBtn}>
            <Text style={styles.backIcon}>←</Text>
          </TouchableOpacity>
          <Text style={styles.navTitle}>{headerTitle}</Text>
          <View style={styles.navSpacer} />
        </View>

        <ScrollView
          contentContainerStyle={styles.content}
          showsVerticalScrollIndicator={false}
          keyboardShouldPersistTaps="handled"
        >
          {/* ── HERO ── */}

          {/* identify mode: labelled identified track */}
          {mode === 'identify' && identified && (
            <View style={styles.section}>
              <Text style={styles.sectionLabel}>IDENTIFIED TRACK</Text>
              <SongCard track={identified} variant="hero" />
            </View>
          )}

          {/* search mode: best match (tappable) */}
          {mode === 'search' && searchResults?.[0] && (
            <View style={styles.section}>
              <Text style={styles.sectionLabel}>BEST MATCH</Text>
              <SongCard
                track={searchResults[0]}
                variant="hero"
                loading={loadingTrackId === searchResults[0].id}
                onPress={() => handleTrackSelect(searchResults[0]!)}
              />
            </View>
          )}

          {/* select mode: hero card, no label */}
          {mode === 'select' && identified && (
            <View style={styles.section}>
              <SongCard track={identified} variant="hero" />
            </View>
          )}

          {/* identify mode: no match */}
          {mode === 'identify' && !identified && (
            <View style={styles.emptyHero}>
              <Text style={styles.emptyIcon}>&#9834;</Text>
              <Text style={styles.emptyText}>Could not identify the track.</Text>
              <Text style={styles.emptySubtext}>
                Try holding your device closer to the speaker.
              </Text>
            </View>
          )}

          {/* ── SIMILAR VIBE ── */}

          {recLoading && (
            <View style={styles.loadingSection}>
              <ActivityIndicator color={colors.accent} />
              <Text style={styles.loadingText}>Finding similar songs...</Text>
            </View>
          )}

          {showNoPreview && (
            <View style={styles.section}>
              <Text style={styles.noPreviewNote}>
                No audio preview available — acoustic recommendations unavailable for this track.
              </Text>
            </View>
          )}

          {!recLoading && displayRecs.length > 0 && (
            <View style={styles.section}>
              <View style={styles.recHeader}>
                <LinearGradient
                  colors={[colors.accentSoft, 'transparent']}
                  style={styles.recHeaderGrad}
                  start={{ x: 0, y: 0 }}
                  end={{ x: 1, y: 0 }}
                />
                <Text style={styles.sectionLabel}>SIMILAR VIBE</Text>
                <Text style={styles.recSubtitle}>Songs that match the beat, mood, and energy</Text>
              </View>
              <View style={styles.cardList}>
                {displayRecs.map((rec, i) => (
                  <SongCard
                    key={rec.id}
                    track={rec}
                    variant="compact"
                    rank={i + 1}
                    onPress={() => openInDeezer(rec)}
                  />
                ))}
              </View>
            </View>
          )}

          {/* ── OTHER SEARCH RESULTS (search mode only) ── */}

          {mode === 'search' && searchResults && searchResults.length > 1 && (
            <View style={styles.section}>
              <Text style={styles.sectionLabel}>OTHER SONGS THAT MATCHED YOUR SEARCH</Text>
              <View style={styles.cardList}>
                {searchResults.slice(1).map(t => (
                  <SongCard
                    key={t.id}
                    track={t}
                    variant="compact"
                    loading={loadingTrackId === t.id}
                    onPress={() => handleTrackSelect(t)}
                  />
                ))}
              </View>
            </View>
          )}
        </ScrollView>
      </SafeAreaView>
    </View>
  );
};

const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: colors.bg,
  },
  safe: {
    flex: 1,
  },
  navbar: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.md,
  },
  backBtn: {
    width: 40,
    height: 40,
    alignItems: 'center',
    justifyContent: 'center',
  },
  navSpacer: {
    width: 40,
  },
  backIcon: {
    fontSize: 24,
    color: colors.textPrimary,
  },
  navTitle: {
    ...typography.headingMd,
    color: colors.textPrimary,
    flex: 1,
    textAlign: 'center',
  },
  content: {
    gap: spacing.xl,
    paddingHorizontal: spacing.md,
    paddingBottom: spacing.xxl,
  },
  section: {
    gap: spacing.md,
  },
  sectionLabel: {
    ...typography.label,
    color: colors.textMuted,
    paddingHorizontal: spacing.xs,
  },
  recHeader: {
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.md,
    gap: 4,
    position: 'relative',
    overflow: 'hidden',
    borderRadius: 12,
  },
  recHeaderGrad: {
    ...StyleSheet.absoluteFillObject,
    borderRadius: 12,
  },
  recSubtitle: {
    ...typography.bodyMd,
    color: colors.textSecondary,
  },
  cardList: {
    gap: spacing.sm,
  },
  loadingSection: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.md,
    paddingHorizontal: spacing.xs,
    paddingVertical: spacing.md,
  },
  loadingText: {
    ...typography.bodyMd,
    color: colors.textMuted,
  },
  noPreviewNote: {
    ...typography.bodySm,
    color: colors.textMuted,
    paddingHorizontal: spacing.xs,
  },
  emptyHero: {
    alignItems: 'center',
    paddingVertical: spacing.xxl,
    gap: spacing.md,
  },
  emptyIcon: {
    fontSize: 48,
    color: colors.textMuted,
  },
  emptyText: {
    ...typography.headingMd,
    color: colors.textSecondary,
  },
  emptySubtext: {
    ...typography.bodyMd,
    color: colors.textMuted,
    textAlign: 'center',
  },
});

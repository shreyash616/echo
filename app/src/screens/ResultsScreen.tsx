import React, { useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  TouchableOpacity,
  StatusBar,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useNavigation, useRoute, RouteProp } from '@react-navigation/native';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { LinearGradient } from 'expo-linear-gradient';

import { SongCard } from '../components/SongCard';
import { colors, spacing, typography } from '../theme';
import type { RootStackParamList } from '../navigation/AppNavigator';
import type { TrackResult } from '../services/api';

type Nav = NativeStackNavigationProp<RootStackParamList, 'Results'>;
type Route = RouteProp<RootStackParamList, 'Results'>;

export const ResultsScreen: React.FC = () => {
  const navigation = useNavigation<Nav>();
  const { params } = useRoute<Route>();
  const { data, mode, query } = params;

  const identified = 'identified' in data ? data.identified : null;
  const searchResults = 'results' in data ? data.results : null;
  const recommendations = data.recommendations;

  const headerTitle = mode === 'identify'
    ? identified ? 'Found it!' : 'No match found'
    : `Results for "${query}"`;

  const renderRecommendation = useCallback(({ item, index }: { item: TrackResult; index: number }) => (
    <SongCard
      track={item}
      variant="compact"
      rank={index + 1}
      onPress={() => {/* TODO: deep detail screen */}}
    />
  ), []);

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
          <View style={{ width: 40 }} />
        </View>

        <FlatList
          data={recommendations}
          keyExtractor={(item) => item.id}
          renderItem={renderRecommendation}
          contentContainerStyle={styles.list}
          showsVerticalScrollIndicator={false}
          ListHeaderComponent={
            <View style={styles.listHeader}>
              {/* Hero: identified song OR first search result */}
              {identified && (
                <View style={styles.heroSection}>
                  <Text style={styles.sectionLabel}>IDENTIFIED TRACK</Text>
                  <SongCard track={identified} variant="hero" />
                </View>
              )}

              {searchResults && searchResults.length > 0 && (
                <View style={styles.heroSection}>
                  <Text style={styles.sectionLabel}>BEST MATCH</Text>
                  <SongCard track={searchResults[0]} variant="hero" />
                  {searchResults.slice(1).map((t) => (
                    <SongCard key={t.id} track={t} variant="compact" />
                  ))}
                </View>
              )}

              {!identified && !searchResults?.length && (
                <View style={styles.emptyHero}>
                  <Text style={styles.emptyIcon}>&#9834;</Text>
                  <Text style={styles.emptyText}>Could not identify the track.</Text>
                  <Text style={styles.emptySubtext}>Try holding your device closer to the speaker.</Text>
                </View>
              )}

              {/* Vibe recommendations header */}
              {recommendations.length > 0 && (
                <View style={styles.recHeader}>
                  <LinearGradient
                    colors={[colors.accentSoft, 'transparent']}
                    style={styles.recHeaderGrad}
                    start={{ x: 0, y: 0 }}
                    end={{ x: 1, y: 0 }}
                  />
                  <Text style={styles.sectionLabel}>SIMILAR VIBE</Text>
                  <Text style={styles.recSubtitle}>
                    Songs that match the beat, mood, and energy
                  </Text>
                </View>
              )}
            </View>
          }
          ItemSeparatorComponent={() => <View style={styles.separator} />}
        />
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
  list: {
    gap: spacing.md,
    paddingBottom: spacing.xxl,
  },
  listHeader: {
    gap: spacing.lg,
  },
  heroSection: {
    gap: spacing.md,
    paddingHorizontal: spacing.md,
  },
  sectionLabel: {
    ...typography.label,
    color: colors.textMuted,
    paddingHorizontal: spacing.xs,
  },
  recHeader: {
    paddingHorizontal: spacing.lg,
    gap: 4,
    position: 'relative',
    overflow: 'hidden',
    paddingVertical: spacing.md,
  },
  recHeaderGrad: {
    position: 'absolute',
    top: 0, left: 0, right: 0, bottom: 0,
    borderRadius: 12,
  },
  recSubtitle: {
    ...typography.bodyMd,
    color: colors.textSecondary,
  },
  separator: {
    height: spacing.sm,
    marginHorizontal: spacing.md,
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

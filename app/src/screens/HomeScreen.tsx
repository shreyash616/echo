import React, { useState, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Alert,
  ActivityIndicator,
  StatusBar,
  Platform,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useNavigation } from '@react-navigation/native';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { LinearGradient } from 'expo-linear-gradient';

import { RecordButton } from '../components/RecordButton';
import { WaveformAnimation } from '../components/WaveformAnimation';
import { SearchBar } from '../components/SearchBar';
import { colors, spacing, typography } from '../theme';
import { requestPermissions, startRecording, stopRecording } from '../services/audio';
import { identifyFromAudio, searchByName } from '../services/api';
import type { RootStackParamList } from '../navigation/AppNavigator';

type Nav = NativeStackNavigationProp<RootStackParamList, 'Home'>;

type Phase = 'idle' | 'recording' | 'identifying' | 'searching';

export const HomeScreen: React.FC = () => {
  const navigation = useNavigation<Nav>();
  const [phase, setPhase] = useState<Phase>('idle');

  const phaseLabel: Record<Phase, string> = {
    idle: 'Tap to identify music',
    recording: 'Listening...',
    identifying: 'Identifying track...',
    searching: 'Searching...',
  };

  const handleRecord = useCallback(async () => {
    if (phase === 'recording') {
      // Stop and identify
      setPhase('identifying');
      try {
        const uri = await stopRecording();
        if (!uri) throw new Error('No recording found');
        const result = await identifyFromAudio(uri);
        navigation.navigate('Results', { data: result, mode: 'identify' });
      } catch (e: unknown) {
        Alert.alert('Could not identify', (e as Error).message);
      } finally {
        setPhase('idle');
      }
      return;
    }

    if (phase !== 'idle') return;

    const granted = await requestPermissions();
    if (!granted) {
      Alert.alert('Permission denied', 'Microphone access is required to identify music.');
      return;
    }

    try {
      await startRecording();
      setPhase('recording');
      // Auto-stop after 30 seconds
      setTimeout(async () => {
        if (phase === 'recording') {
          setPhase('identifying');
          const uri = await stopRecording();
          if (!uri) { setPhase('idle'); return; }
          try {
            const result = await identifyFromAudio(uri);
            navigation.navigate('Results', { data: result, mode: 'identify' });
          } catch (e: unknown) {
            Alert.alert('Could not identify', (e as Error).message);
          } finally {
            setPhase('idle');
          }
        }
      }, 30000);
    } catch (e: unknown) {
      Alert.alert('Recording failed', (e as Error).message);
      setPhase('idle');
    }
  }, [phase, navigation]);

  const handleSearch = useCallback(async (query: string) => {
    setPhase('searching');
    try {
      const result = await searchByName(query);
      navigation.navigate('Results', { data: result, mode: 'search', query });
    } catch (e: unknown) {
      Alert.alert('Search failed', (e as Error).message);
    } finally {
      setPhase('idle');
    }
  }, [navigation]);

  const isActive = phase === 'recording';
  const isBusy = phase === 'identifying' || phase === 'searching';

  return (
    <View style={styles.root}>
      <StatusBar barStyle="light-content" />
      {/* Ambient gradient blob */}
      <LinearGradient
        colors={['rgba(139,92,246,0.18)', 'transparent']}
        style={styles.ambientGlow}
        start={{ x: 0.5, y: 0 }}
        end={{ x: 0.5, y: 1 }}
      />

      <SafeAreaView style={styles.safe}>
        <ScrollView
          contentContainerStyle={styles.content}
          keyboardShouldPersistTaps="handled"
          showsVerticalScrollIndicator={false}
        >
          {/* Header */}
          <View style={styles.header}>
            <Text style={styles.wordmark}>echo</Text>
            <Text style={styles.tagline}>Discover your sound</Text>
          </View>

          {/* Record section */}
          <View style={styles.recordSection}>
            <WaveformAnimation active={isActive} />

            <View style={styles.buttonRow}>
              {isBusy ? (
                <ActivityIndicator size="large" color={colors.accent} />
              ) : (
                <RecordButton recording={isActive} onPress={handleRecord} />
              )}
            </View>

            <Text style={styles.phaseLabel}>{phaseLabel[phase]}</Text>
            {isActive && (
              <Text style={styles.tapToStop}>Tap again to stop</Text>
            )}
          </View>

          {/* Divider */}
          <View style={styles.dividerRow}>
            <View style={styles.dividerLine} />
            <Text style={styles.dividerText}>or search</Text>
            <View style={styles.dividerLine} />
          </View>

          {/* Search */}
          <View style={styles.searchRow}>
            <SearchBar onSearch={handleSearch} loading={phase === 'searching'} />
          </View>

          {/* Footer hint */}
          <Text style={styles.hint}>
            Powered by your custom ML model trained on 100k tracks
          </Text>
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
  ambientGlow: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    height: 320,
  },
  content: {
    flexGrow: 1,
    paddingHorizontal: spacing.lg,
    paddingBottom: spacing.xxl,
    gap: spacing.xl,
  },
  header: {
    marginTop: spacing.xl,
    gap: 4,
  },
  wordmark: {
    fontSize: 38,
    fontWeight: '800',
    color: colors.textPrimary,
    letterSpacing: -2,
  },
  tagline: {
    ...typography.bodyLg,
    color: colors.textMuted,
  },
  recordSection: {
    alignItems: 'center',
    gap: spacing.lg,
    paddingVertical: spacing.xl,
  },
  buttonRow: {
    width: 88,
    height: 88,
    alignItems: 'center',
    justifyContent: 'center',
  },
  phaseLabel: {
    ...typography.bodyLg,
    color: colors.textSecondary,
    textAlign: 'center',
  },
  tapToStop: {
    ...typography.bodySm,
    color: colors.textMuted,
    textAlign: 'center',
  },
  dividerRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.md,
  },
  dividerLine: {
    flex: 1,
    height: 1,
    backgroundColor: colors.border,
  },
  dividerText: {
    ...typography.bodySm,
    color: colors.textMuted,
  },
  searchRow: {
    gap: spacing.sm,
  },
  hint: {
    ...typography.bodySm,
    color: colors.textMuted,
    textAlign: 'center',
    marginTop: 'auto',
  },
});

import React, { useState, useCallback, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Alert,
  ActivityIndicator,
  StatusBar,
  Animated,
  Easing,
  KeyboardAvoidingView,
  Platform,
} from 'react-native';
import { SafeAreaView, useSafeAreaInsets } from 'react-native-safe-area-context';
import { useNavigation, useFocusEffect } from '@react-navigation/native';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { LinearGradient } from 'expo-linear-gradient';
import { BlurView } from 'expo-blur';

import { RecordButton } from '../components/RecordButton';
import { WaveformAnimation } from '../components/WaveformAnimation';
import { SearchBar } from '../components/SearchBar';
import type { SearchBarHandle } from '../components/SearchBar';
import { colors, radius, spacing, typography } from '../theme';
import { requestPermissions, startRecording, stopRecording } from '../services/audio';
import { identifyFromAudio, searchByName } from '../services/api';
import type { RootStackParamList } from '../navigation/AppNavigator';

type Nav = NativeStackNavigationProp<RootStackParamList, 'Home'>;
type Phase = 'idle' | 'recording' | 'identifying' | 'searching';

export const HomeScreen: React.FC = () => {
  const navigation = useNavigation<Nav>();
  const [phase, setPhase] = useState<Phase>('idle');
  const phaseRef = useRef<Phase>('idle');
  const setPhaseSync = (p: Phase) => {
    phaseRef.current = p;
    setPhase(p);
  };
  const searchAnim = useRef(new Animated.Value(0)).current;
  const recordModeAnim = useRef(new Animated.Value(0)).current;
  const searchBarRef = useRef<SearchBarHandle>(null);

  // Reset all UI state when navigating back to this screen
  useFocusEffect(
    useCallback(() => {
      searchAnim.setValue(0);
      recordModeAnim.setValue(0);
      searchBarRef.current?.reset();
    }, [searchAnim, recordModeAnim]),
  );

  // Measured distance from top of header to top of search pill (header height + gap)
  const [searchOffset, setSearchOffset] = useState(0);
  // Measured Y position of the record section within scroll content
  const [recordY, setRecordY] = useState(0);
  const { top: safeTop } = useSafeAreaInsets();

  const openSearch = useCallback(() => {
    Animated.timing(searchAnim, {
      toValue: 1,
      duration: 260,
      easing: Easing.out(Easing.quad),
      useNativeDriver: true,
    }).start();
  }, [searchAnim]);

  const closeSearch = useCallback(() => {
    Animated.timing(searchAnim, {
      toValue: 0,
      duration: 220,
      easing: Easing.in(Easing.quad),
      useNativeDriver: true,
    }).start();
  }, [searchAnim]);

  const openRecord = useCallback(() => {
    Animated.timing(recordModeAnim, {
      toValue: 1,
      duration: 260,
      easing: Easing.out(Easing.quad),
      useNativeDriver: false,
    }).start();
  }, [recordModeAnim]);

  const closeRecord = useCallback(() => {
    Animated.timing(recordModeAnim, {
      toValue: 0,
      duration: 220,
      easing: Easing.in(Easing.quad),
      useNativeDriver: false,
    }).start();
  }, [recordModeAnim]);

  const phaseLabel: Record<Phase, string> = {
    idle: 'Tap to identify music',
    recording: 'Listening...',
    identifying: 'Identifying track...',
    searching: 'Searching...',
  };

  const handleRecord = useCallback(async () => {
    if (phase === 'recording') {
      closeRecord();
      setPhaseSync('identifying');
      try {
        const uri = await stopRecording();
        if (!uri) throw new Error('No recording found');
        const result = await identifyFromAudio(uri);
        navigation.navigate('Results', { data: result, mode: 'identify' });
      } catch (e: unknown) {
        Alert.alert('Could not identify', (e as Error).message);
      } finally {
        setPhaseSync('idle');
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
      openRecord();
      setPhaseSync('recording');
      setTimeout(async () => {
        if (phaseRef.current === 'recording') {
          closeRecord();
          setPhaseSync('identifying');
          const uri = await stopRecording();
          if (!uri) {
            setPhaseSync('idle');
            return;
          }
          try {
            const result = await identifyFromAudio(uri);
            navigation.navigate('Results', { data: result, mode: 'identify' });
          } catch (e: unknown) {
            Alert.alert('Could not identify', (e as Error).message);
          } finally {
            setPhaseSync('idle');
          }
        }
      }, 30000);
    } catch (e: unknown) {
      Alert.alert('Recording failed', (e as Error).message);
      setPhaseSync('idle');
    }
  }, [phase, navigation, openRecord, closeRecord]);

  const handleSearch = useCallback(
    async (query: string) => {
      setPhaseSync('searching');
      try {
        const result = await searchByName(query);
        closeSearch();
        navigation.navigate('Results', { data: result, mode: 'search', query });
      } catch (e: unknown) {
        Alert.alert('Search failed', (e as Error).message);
      } finally {
        setPhaseSync('idle');
      }
    },
    [navigation, closeSearch],
  );

  const isActive = phase === 'recording';
  const isBusy = phase === 'identifying' || phase === 'searching';

  const headerAnim = {
    opacity: searchAnim.interpolate({ inputRange: [0, 0.5], outputRange: [1, 0] }),
    transform: [
      { translateY: searchAnim.interpolate({ inputRange: [0, 1], outputRange: [0, -12] }) },
    ],
  };
  // Search pill flows up by the measured offset to anchor at the top
  const searchSlideAnim = {
    transform: [
      {
        translateY: searchAnim.interpolate({ inputRange: [0, 1], outputRange: [0, -searchOffset] }),
      },
    ],
  };
  const dividerSearchAnim = {
    opacity: searchAnim.interpolate({ inputRange: [0, 0.3], outputRange: [1, 0] }),
  };
  const recordAnim = {
    opacity: searchAnim.interpolate({ inputRange: [0, 0.6], outputRange: [1, 0] }),
    transform: [
      { translateY: searchAnim.interpolate({ inputRange: [0, 1], outputRange: [0, 56] }) },
    ],
  };

  // Record-mode animations (JS driver — needed for layout props)
  const recordModeHeaderAnim = {
    opacity: recordModeAnim.interpolate({ inputRange: [0, 0.5], outputRange: [1, 0] }),
    transform: [
      { translateY: recordModeAnim.interpolate({ inputRange: [0, 1], outputRange: [0, -12] }) },
    ],
  };
  const recordModePillAnim = {
    opacity: recordModeAnim.interpolate({ inputRange: [0, 0.6], outputRange: [1, 0] }),
    transform: [
      { translateY: recordModeAnim.interpolate({ inputRange: [0, 1], outputRange: [0, -48] }) },
    ],
  };
  const recordModeDividerAnim = {
    opacity: recordModeAnim.interpolate({ inputRange: [0, 0.4], outputRange: [1, 0] }),
  };
  const recordSectionGap = recordModeAnim.interpolate({
    inputRange: [0, 1],
    outputRange: [spacing.lg, spacing.xxl],
  });
  // Translate the record section up to sit near the top of the safe area
  const recordSectionTranslate = recordModeAnim.interpolate({
    inputRange: [0, 1],
    outputRange: [0, -(recordY - safeTop - spacing.xl)],
  });

  return (
    <View style={styles.root}>
      <StatusBar barStyle="light-content" />
      <LinearGradient
        colors={['rgba(139,92,246,0.18)', 'transparent']}
        style={styles.ambientGlow}
        start={{ x: 0.5, y: 0 }}
        end={{ x: 0.5, y: 1 }}
      />

      <SafeAreaView style={styles.safe}>
        <KeyboardAvoidingView
          style={styles.kav}
          behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        >
          <ScrollView
            contentContainerStyle={styles.content}
            keyboardShouldPersistTaps="handled"
            showsVerticalScrollIndicator={false}
          >
            {/* Header — fades on search open (native driver) and record mode (JS driver) */}
            <Animated.View style={recordModeHeaderAnim}>
              <Animated.View
                style={[styles.header, headerAnim]}
                onLayout={e => setSearchOffset(e.nativeEvent.layout.height + spacing.xl)}
              >
                <Text style={styles.wordmark}>echo</Text>
                <Text style={styles.tagline}>Discover your sound</Text>
              </Animated.View>
            </Animated.View>

            {/* Search bar — slides to top when search opens; fades up when recording */}
            <Animated.View style={searchSlideAnim}>
              <Animated.View style={recordModePillAnim}>
                <SearchBar
                  ref={searchBarRef}
                  onSearch={handleSearch}
                  onOpen={openSearch}
                  onClose={closeSearch}
                  loading={phase === 'searching'}
                />
              </Animated.View>
            </Animated.View>

            {/* Divider fades on search open (native driver) and record mode (JS driver) */}
            <Animated.View style={dividerSearchAnim}>
              <Animated.View style={recordModeDividerAnim}>
                <View style={styles.dividerRow}>
                  <View style={styles.dividerLine} />
                  <View style={styles.dividerPill}>
                    {Platform.OS === 'ios' && (
                      <BlurView intensity={22} tint="dark" style={StyleSheet.absoluteFill} />
                    )}
                    <View style={[StyleSheet.absoluteFill, styles.dividerPillOverlay]} />
                    <Text style={styles.dividerText}>or</Text>
                  </View>
                  <View style={styles.dividerLine} />
                </View>
              </Animated.View>
            </Animated.View>

            {/* Record section — fades on search open; slides to top + expands on record mode */}
            <Animated.View style={recordAnim} onLayout={e => setRecordY(e.nativeEvent.layout.y)}>
              <Animated.View
                style={[
                  styles.recordSection,
                  { gap: recordSectionGap, transform: [{ translateY: recordSectionTranslate }] },
                ]}
              >
                <WaveformAnimation active={isActive} />
                <View style={styles.buttonRow}>
                  {isBusy ? (
                    <ActivityIndicator size="large" color={colors.accent} />
                  ) : (
                    <RecordButton recording={isActive} onPress={handleRecord} />
                  )}
                </View>
                <View style={styles.phasePill}>
                  {Platform.OS === 'ios' && (
                    <BlurView intensity={18} tint="dark" style={StyleSheet.absoluteFill} />
                  )}
                  <View style={[StyleSheet.absoluteFill, styles.phasePillOverlay]} />
                  <Text style={styles.phaseLabel}>{phaseLabel[phase]}</Text>
                </View>
                {isActive && <Text style={styles.tapToStop}>Tap again to stop</Text>}
              </Animated.View>
            </Animated.View>
          </ScrollView>
        </KeyboardAvoidingView>
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
  kav: {
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
  dividerRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.md,
  },
  dividerLine: {
    flex: 1,
    height: 1,
    backgroundColor: 'rgba(255,255,255,0.07)',
  },
  dividerPill: {
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.xs,
    borderRadius: radius.full,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.1)',
    overflow: 'hidden',
    backgroundColor: Platform.OS === 'android' ? 'rgba(20,20,26,0.9)' : 'transparent',
  },
  dividerPillOverlay: {
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderRadius: radius.full,
  },
  dividerText: {
    ...typography.bodySm,
    color: colors.textMuted,
  },
  recordSection: {
    alignItems: 'center',
    paddingVertical: spacing.xl,
  },
  buttonRow: {
    width: 88,
    height: 88,
    alignItems: 'center',
    justifyContent: 'center',
  },
  phasePill: {
    alignSelf: 'center',
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.sm,
    borderRadius: radius.full,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.09)',
    overflow: 'hidden',
    backgroundColor: Platform.OS === 'android' ? 'rgba(20,20,26,0.88)' : 'transparent',
  },
  phasePillOverlay: {
    backgroundColor: 'rgba(255,255,255,0.04)',
    borderRadius: radius.full,
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
});

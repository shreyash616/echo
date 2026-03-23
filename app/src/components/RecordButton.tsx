import React, { useEffect } from 'react';
import { TouchableOpacity, StyleSheet, View } from 'react-native';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withRepeat,
  withTiming,
  withSequence,
  Easing,
  cancelAnimation,
} from 'react-native-reanimated';
import { LinearGradient } from 'expo-linear-gradient';
import * as Haptics from 'expo-haptics';
import { colors } from '../theme';

interface Props {
  recording: boolean;
  onPress: () => void;
  size?: number;
}

export const RecordButton: React.FC<Props> = ({ recording, onPress, size = 88 }) => {
  const scale = useSharedValue(1);
  const ringOpacity = useSharedValue(0);
  const ringScale = useSharedValue(1);

  useEffect(() => {
    if (recording) {
      scale.value = withRepeat(
        withSequence(
          withTiming(0.94, { duration: 600, easing: Easing.inOut(Easing.ease) }),
          withTiming(1, { duration: 600, easing: Easing.inOut(Easing.ease) }),
        ),
        -1,
        false,
      );
      ringOpacity.value = withRepeat(
        withSequence(withTiming(0.6, { duration: 600 }), withTiming(0, { duration: 800 })),
        -1,
        false,
      );
      ringScale.value = withRepeat(
        withSequence(
          withTiming(1, { duration: 0 }),
          withTiming(1.7, { duration: 1400, easing: Easing.out(Easing.ease) }),
        ),
        -1,
        false,
      );
    } else {
      cancelAnimation(scale);
      cancelAnimation(ringOpacity);
      cancelAnimation(ringScale);
      scale.value = withTiming(1, { duration: 200 });
      ringOpacity.value = withTiming(0, { duration: 200 });
      ringScale.value = withTiming(1, { duration: 200 });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [recording]);

  const buttonStyle = useAnimatedStyle(() => ({
    transform: [{ scale: scale.value }],
  }));

  const ringStyle = useAnimatedStyle(() => ({
    opacity: ringOpacity.value,
    transform: [{ scale: ringScale.value }],
  }));

  const handlePress = async () => {
    await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Heavy);
    onPress();
  };

  return (
    <TouchableOpacity onPress={handlePress} activeOpacity={0.9}>
      <View style={[styles.container, { width: size, height: size }]}>
        {/* Pulse ring */}
        <Animated.View
          style={[styles.ring, ringStyle, { width: size, height: size, borderRadius: size / 2 }]}
        />
        {/* Button */}
        <Animated.View style={[buttonStyle, { width: size, height: size }]}>
          <LinearGradient
            colors={recording ? ['#EF4444', '#DC2626'] : ['#A78BFA', '#8B5CF6']}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 1 }}
            style={[styles.button, { width: size, height: size, borderRadius: size / 2 }]}
          >
            {/* Inner icon: mic bars or stop square */}
            <View style={styles.iconContainer}>
              {recording ? <View style={styles.stopIcon} /> : <MicIcon size={size * 0.36} />}
            </View>
          </LinearGradient>
        </Animated.View>
      </View>
    </TouchableOpacity>
  );
};

const MicIcon: React.FC<{ size: number }> = ({ size }) => (
  <View style={styles.micWrapper}>
    {/* Simple mic shape using views */}
    <View
      style={[
        styles.micBody,
        { width: size * 0.5, height: size * 0.65, borderRadius: size * 0.25 },
      ]}
    />
    <View style={[styles.micBase, { width: size * 0.8, height: size * 0.1 }]} />
    <View style={[styles.micStand, { width: size * 0.12, height: size * 0.15 }]} />
  </View>
);

const styles = StyleSheet.create({
  container: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  button: {
    shadowColor: colors.accent,
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.5,
    shadowRadius: 20,
    elevation: 10,
  },
  ring: {
    position: 'absolute',
    borderWidth: 2,
    borderColor: colors.accent,
  },
  iconContainer: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  stopIcon: {
    width: 22,
    height: 22,
    backgroundColor: '#fff',
    borderRadius: 4,
  },
  micBody: {
    backgroundColor: '#fff',
  },
  micWrapper: {
    alignItems: 'center',
    gap: 2,
  },
  micBase: {
    backgroundColor: '#fff',
    borderRadius: 2,
    opacity: 0.9,
  },
  micStand: {
    backgroundColor: '#fff',
    opacity: 0.9,
  },
});

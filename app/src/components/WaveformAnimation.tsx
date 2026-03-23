import React, { useEffect } from 'react';
import { View, StyleSheet } from 'react-native';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withRepeat,
  withTiming,
  withDelay,
  Easing,
  cancelAnimation,
} from 'react-native-reanimated';
import { colors } from '../theme';

const BAR_COUNT = 32;
const BAR_MAX_HEIGHT = 48;
const BAR_MIN_HEIGHT = 4;

interface Props {
  active: boolean;
  color?: string;
}

const Bar: React.FC<{ index: number; active: boolean; color: string }> = ({
  index,
  active,
  color,
}) => {
  const height = useSharedValue(BAR_MIN_HEIGHT);

  useEffect(() => {
    if (active) {
      const targetH = BAR_MIN_HEIGHT + Math.random() * (BAR_MAX_HEIGHT - BAR_MIN_HEIGHT);
      const duration = 300 + Math.random() * 400;
      height.value = withDelay(
        index * 18,
        withRepeat(withTiming(targetH, { duration, easing: Easing.inOut(Easing.sin) }), -1, true),
      );
    } else {
      cancelAnimation(height);
      height.value = withTiming(BAR_MIN_HEIGHT, { duration: 300 });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active]);

  const style = useAnimatedStyle(() => ({
    height: height.value,
  }));

  return <Animated.View style={[styles.bar, style, { backgroundColor: color }]} />;
};

export const WaveformAnimation: React.FC<Props> = ({ active, color = colors.accent }) => {
  return (
    <View style={styles.container}>
      {Array.from({ length: BAR_COUNT }).map((_, i) => (
        <Bar key={i} index={i} active={active} color={color} />
      ))}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    height: BAR_MAX_HEIGHT + 8,
  },
  bar: {
    width: 3,
    borderRadius: 2,
  },
});

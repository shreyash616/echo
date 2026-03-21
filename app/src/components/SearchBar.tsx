import React, { useRef, useState } from 'react';
import {
  View,
  TextInput,
  StyleSheet,
  TouchableOpacity,
  Text,
  Animated as RNAnimated,
} from 'react-native';
import { colors, radius, spacing, typography } from '../theme';

interface Props {
  onSearch: (query: string) => void;
  loading?: boolean;
}

export const SearchBar: React.FC<Props> = ({ onSearch, loading }) => {
  const [value, setValue] = useState('');
  const [focused, setFocused] = useState(false);
  const borderOpacity = useRef(new RNAnimated.Value(0)).current;

  const handleFocus = () => {
    setFocused(true);
    RNAnimated.timing(borderOpacity, { toValue: 1, duration: 200, useNativeDriver: false }).start();
  };

  const handleBlur = () => {
    setFocused(false);
    RNAnimated.timing(borderOpacity, { toValue: 0, duration: 200, useNativeDriver: false }).start();
  };

  const borderColor = borderOpacity.interpolate({
    inputRange: [0, 1],
    outputRange: [colors.border, colors.borderFocused],
  });

  const submit = () => {
    if (value.trim()) onSearch(value.trim());
  };

  return (
    <RNAnimated.View style={[styles.container, { borderColor }]}>
      <Text style={styles.icon}>&#9836;</Text>
      <TextInput
        style={styles.input}
        placeholder="Search song or artist..."
        placeholderTextColor={colors.textMuted}
        value={value}
        onChangeText={setValue}
        onFocus={handleFocus}
        onBlur={handleBlur}
        onSubmitEditing={submit}
        returnKeyType="search"
        selectionColor={colors.accent}
      />
      {value.length > 0 && (
        <TouchableOpacity onPress={submit} style={styles.searchBtn} activeOpacity={0.7}>
          <Text style={styles.searchBtnText}>{loading ? '...' : 'Go'}</Text>
        </TouchableOpacity>
      )}
    </RNAnimated.View>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.bgInput,
    borderRadius: radius.full,
    borderWidth: 1,
    paddingHorizontal: spacing.md,
    height: 52,
    gap: spacing.sm,
  },
  icon: {
    fontSize: 16,
    color: colors.textMuted,
  },
  input: {
    flex: 1,
    ...typography.bodyLg,
    color: colors.textPrimary,
  },
  searchBtn: {
    backgroundColor: colors.accent,
    borderRadius: radius.full,
    paddingHorizontal: spacing.md,
    paddingVertical: 6,
  },
  searchBtnText: {
    ...typography.bodySm,
    color: '#fff',
    fontWeight: '700',
  },
});

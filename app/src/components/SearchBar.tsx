import React, { useRef, useState, useImperativeHandle, forwardRef } from 'react';
import {
  View,
  TextInput,
  StyleSheet,
  TouchableOpacity,
  Text,
  Animated,
  ActivityIndicator,
} from 'react-native';
import { colors, radius, spacing, typography } from '../theme';

interface Props {
  onSearch: (query: string) => void;
  onOpen?: () => void;
  onClose?: () => void;
  loading?: boolean;
}

export type SearchBarHandle = { reset: () => void };

export const SearchBar = forwardRef<SearchBarHandle, Props>(
function SearchBar({ onSearch, onOpen, onClose, loading }, ref) {
  const [value, setValue] = useState('');
  const [isOpen, setIsOpen] = useState(false);
  const inputRef = useRef<TextInput>(null);

  // useNativeDriver: false required for layout props (width, borderColor)
  const openAnim = useRef(new Animated.Value(0)).current;

  useImperativeHandle(ref, () => ({
    reset: () => {
      inputRef.current?.blur();
      openAnim.setValue(0);
      setValue('');
      setIsOpen(false);
      onClose?.();
    },
  }));

  const open = () => {
    setIsOpen(true);
    onOpen?.();
    Animated.timing(openAnim, {
      toValue: 1,
      duration: 220,
      useNativeDriver: false,
    }).start(() => inputRef.current?.focus());
  };

  const close = () => {
    inputRef.current?.blur();
    setValue('');
    Animated.timing(openAnim, {
      toValue: 0,
      duration: 180,
      useNativeDriver: false,
    }).start(() => {
      setIsOpen(false);
      onClose?.();
    });
  };

  const submit = () => {
    if (value.trim()) onSearch(value.trim());
  };

  const backBtnWidth = openAnim.interpolate({
    inputRange: [0, 1],
    outputRange: [0, 24],
  });
  const borderColor = openAnim.interpolate({
    inputRange: [0, 1],
    outputRange: [colors.border, colors.borderFocused],
  });

  return (
    <Animated.View style={[styles.container, { borderColor }]}>
      {/* Back button — slides in by expanding width */}
      <Animated.View style={{ width: backBtnWidth, overflow: 'hidden' }}>
        <TouchableOpacity onPress={close} style={styles.cancelBtn} activeOpacity={0.7}>
          <Text style={styles.cancelArrow}>←</Text>
        </TouchableOpacity>
      </Animated.View>

      <Text style={styles.icon}>&#9836;</Text>

      <View style={styles.inputWrap} pointerEvents={isOpen ? 'auto' : 'none'}>
        <TextInput
          ref={inputRef}
          style={styles.input}
          placeholder="Search song or artist..."
          placeholderTextColor={colors.textMuted}
          value={value}
          onChangeText={setValue}
          onSubmitEditing={submit}
          returnKeyType="search"
          selectionColor={colors.accent}
          editable={isOpen}
        />
      </View>

      {/* Invisible tap target over the whole pill when closed */}
      {!isOpen && (
        <TouchableOpacity
          style={StyleSheet.absoluteFill}
          onPress={open}
          activeOpacity={0.7}
        />
      )}

      {isOpen && value.length > 0 && (
        <TouchableOpacity onPress={submit} style={styles.searchBtn} activeOpacity={0.7}>
          {loading ? (
            <ActivityIndicator size="small" color="#fff" />
          ) : (
            <Text style={styles.searchBtnText}>Find</Text>
          )}
        </TouchableOpacity>
      )}
    </Animated.View>
  );
});

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
  cancelBtn: {
    width: 22,
    height: 32,
    alignItems: 'center',
    justifyContent: 'center',
  },
  cancelArrow: {
    fontSize: 20,
    color: colors.textPrimary,
  },
  icon: {
    fontSize: 16,
    color: colors.textMuted,
  },
  inputWrap: {
    flex: 1,
  },
  input: {
    ...typography.bodyLg,
    color: colors.textPrimary,
  },
  searchBtn: {
    backgroundColor: colors.accent,
    borderRadius: radius.full,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.xs,
    minWidth: 52,
    alignItems: 'center',
  },
  searchBtnText: {
    ...typography.bodySm,
    color: '#fff',
    fontWeight: '700',
  },
});

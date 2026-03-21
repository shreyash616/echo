import { Audio } from 'expo-av';

let recording: Audio.Recording | null = null;

export async function requestPermissions(): Promise<boolean> {
  const { status } = await Audio.requestPermissionsAsync();
  return status === 'granted';
}

export async function startRecording(): Promise<void> {
  await Audio.setAudioModeAsync({
    allowsRecordingIOS: true,
    playsInSilentModeIOS: true,
  });

  const { recording: rec } = await Audio.Recording.createAsync(
    Audio.RecordingOptionsPresets.HIGH_QUALITY
  );
  recording = rec;
}

export async function stopRecording(): Promise<string | null> {
  if (!recording) return null;

  await recording.stopAndUnloadAsync();
  const uri = recording.getURI();
  recording = null;

  await Audio.setAudioModeAsync({ allowsRecordingIOS: false });
  return uri ?? null;
}

export function isRecording(): boolean {
  return recording !== null;
}

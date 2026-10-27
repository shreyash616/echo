const BASE_URL = 'https://shreyash616-echo-backend.hf.space';

export interface TrackResult {
  id: string;
  key?: string;
  title: string;
  artist: string;
  album: string;
  albumArtUrl: string;
  previewUrl?: string;
  durationMs: number;
  matchScore?: number; // 0-1 similarity score from ML model
  vibes: string[]; // anchor vibes
  theme?: string; // canonical human emotional theme (Section 12.3)
  matchReasons?: string[];
  languageAffinity?: number;
  bpm: number;
  felt_bpm?: number;
  key: string;
  energy: number; // 0-1
  valence: number; // 0-1
}

export interface IdentifyResponse {
  identified: TrackResult | null;
  recommendations: TrackResult[];
}

export interface SearchResponse {
  results: TrackResult[];
  recommendations: TrackResult[];
}

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(err.detail ?? `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export async function identifyFromAudio(audioUri: string): Promise<IdentifyResponse> {
  const formData = new FormData();
  formData.append('audio', {
    uri: audioUri,
    type: 'audio/m4a',
    name: 'snippet.m4a',
  } as unknown as Blob);

  const res = await fetch(`${BASE_URL}/api/identify`, {
    method: 'POST',
    body: formData,
  });
  return handleResponse<IdentifyResponse>(res);
}

export async function searchByName(query: string): Promise<SearchResponse> {
  const params = new URLSearchParams({ q: query });
  const res = await fetch(`${BASE_URL}/api/search?${params}`);
  return handleResponse<SearchResponse>(res);
}

export async function getRecommendations(trackId: string, trackKey?: string): Promise<TrackResult[]> {
  const url = trackKey
    ? `${BASE_URL}/api/recommendations/${encodeURIComponent(trackId)}?key=${encodeURIComponent(trackKey)}`
    : `${BASE_URL}/api/recommendations/${encodeURIComponent(trackId)}`;
  const res = await fetch(url);
  const data = await handleResponse<{ recommendations: TrackResult[] } | TrackResult[]>(res);
  return Array.isArray(data) ? data : (data.recommendations || []);
}

export async function getTrack(trackId: string, trackKey?: string): Promise<TrackResult> {
  const url = trackKey
    ? `${BASE_URL}/api/track/${encodeURIComponent(trackId)}?key=${encodeURIComponent(trackKey)}`
    : `${BASE_URL}/api/track/${encodeURIComponent(trackId)}`;
  const res = await fetch(url);
  return handleResponse<TrackResult>(res);
}

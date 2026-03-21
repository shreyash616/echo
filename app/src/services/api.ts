const BASE_URL = 'http://localhost:8000'; // Change to your backend IP when testing on device

export interface TrackResult {
  id: string;
  title: string;
  artist: string;
  album: string;
  albumArtUrl: string;
  previewUrl?: string;
  durationMs: number;
  matchScore?: number; // 0-1 similarity score from ML model
  vibes: string[];    // e.g. ["dark", "energetic", "melodic"]
  bpm: number;
  key: string;
  energy: number;   // 0-1
  valence: number;  // 0-1 (happiness)
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

export async function getRecommendations(trackId: string): Promise<TrackResult[]> {
  const res = await fetch(`${BASE_URL}/api/recommendations/${trackId}`);
  return handleResponse<TrackResult[]>(res);
}

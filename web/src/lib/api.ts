import type { ClientConfig, IdentifyResponse, Track } from "./types";

// Every call is same-origin: in production FastAPI serves this app, in `next dev` next.config.ts proxies /api.

// The old page's slider said 0.72, but a range input with min 0.50 and step 0.05 snaps that to 0.70, which
// is what it sent (and the backend's own default)
export const DEFAULT_DIVERSITY = 0.7;
export const FALLBACK_ART = "https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4";
export const MIC_RECORDING_ID = "mic_recording";

// Canonical "artist:title" key the backend matches the catalogue on
export function trackKey(t: Pick<Track, "key" | "artist" | "title">): string {
  if (t.key && t.key.includes(":")) return t.key;
  return t.artist && t.title ? `${t.artist.toLowerCase().trim()}:${t.title.toLowerCase().trim()}` : "";
}

export function decadeOf(t: Track): string {
  return t.decade || (t.year ? `${Math.floor(t.year / 10) * 10}s` : "");
}

export function songHref(t: Pick<Track, "id" | "key" | "artist" | "title">, diversity?: number): string {
  const params = new URLSearchParams({ id: String(t.id) });
  const key = trackKey(t);
  if (key) params.set("key", key);
  if (diversity !== undefined && diversity !== DEFAULT_DIVERSITY) params.set("d", diversity.toFixed(2));
  return `/song/?${params}`;
}

// "A, B & C" -> "A", the same rule as _primary_artist_name in backend/app/api/routes.py
export function primaryArtist(artist: string): string {
  const parts = artist.split(/\s*(?:,|&|\bfeat\.?|\bft\.?|\bx\b|\/)\s*/i);
  return (parts[0] || artist).trim();
}

export function artistHref(artist: string): string {
  return `/artist/?${new URLSearchParams({ name: primaryArtist(artist) })}`;
}

export function searchHref(q: string): string {
  return `/search/?${new URLSearchParams({ q })}`;
}

export class HttpError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function getJson<T>(url: string, what: string, signal?: AbortSignal): Promise<T> {
  const res = await fetch(url, { signal });
  if (!res.ok) throw new HttpError(res.status, `${what} failed (${res.status})`);
  return res.json();
}

export async function searchTracks(q: string, limit: number, signal?: AbortSignal): Promise<Track[]> {
  const data = await getJson<{ results?: Track[] }>(
    `/api/search?q=${encodeURIComponent(q)}&limit=${limit}`, "Search", signal);
  return (data.results || []).slice(0, limit);
}

export async function getTrack(id: string, key: string, signal?: AbortSignal): Promise<Track> {
  const keyParam = key ? `?key=${encodeURIComponent(key)}` : "";
  return getJson<Track>(`/api/track/${encodeURIComponent(id)}${keyParam}`, "Track lookup", signal);
}

export async function getRecommendations(
  id: string, key: string, diversity: number, signal?: AbortSignal,
): Promise<Track[]> {
  const keyParam = key ? `&key=${encodeURIComponent(key)}` : "";
  const data = await getJson<Track[] | { recommendations?: Track[] }>(
    `/api/recommendations/${encodeURIComponent(id)}?diversity=${diversity}${keyParam}`, "Recommendations", signal);
  return Array.isArray(data) ? data : data.recommendations || [];
}

export async function identify(blob: Blob): Promise<IdentifyResponse> {
  const formData = new FormData();
  const ext = blob.type.includes("mp4") ? "mp4" : blob.type.includes("ogg") ? "ogg" : "webm";
  formData.append("audio", blob, `recording.${ext}`);
  formData.append("file", blob, `recording.${ext}`);
  const res = await fetch("/api/identify", { method: "POST", body: formData });
  if (!res.ok) throw new Error("Identification failed (" + res.status + ")");
  return res.json();
}

export async function getConfig(): Promise<ClientConfig> {
  return getJson<ClientConfig>("/api/config", "Config");
}

// Songs seen in search results and song cards, so a song page can show its title while it loads and
// fall back to the card's data when /api/track fails, as the old page did.
const seen = new Map<string, Track>();

export function remember(tracks: Track[]): void {
  for (const t of tracks) if (t.id) seen.set(String(t.id), t);
}

export function recall(id: string): Track | undefined {
  return seen.get(id);
}

import { pageUrl } from "./api";
import type { Track } from "./types";

// Spotify playlist export, ported from backend/static/index.html (with the d150967 fix: /me/playlists,
// /playlists/{id}/items, every response checked). PKCE per RFC 7636. localStorage keys are the old page's,
// so a login started on either page completes on this one.

export const LS = {
  token: "spotify_token",
  refreshToken: "spotify_refresh_token",
  verifier: "spotify_code_verifier",
  clientId: "spotify_client_id",
  redirectUri: "spotify_redirect_uri",
  usedRedirectUri: "spotify_used_redirect_uri",
  confirmed: "spotify_auth_confirmed",
  pending: "pending_spotify_playlist",
} as const;

export interface PlaylistTrack {
  title: string;
  artist: string;
}

export interface PendingPlaylist {
  seed: string;
  tracks: PlaylistTrack[];
  // Page the login started from, reopened after Spotify sends the browser back to /
  returnTo?: string;
}

export function lsGet(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

export function lsSet(key: string, value: string): void {
  try {
    localStorage.setItem(key, value);
  } catch {}
}

export function lsRemove(key: string): void {
  try {
    localStorage.removeItem(key);
  } catch {}
}

// Spotify's redirect URI is the site root (the URI registered for the Space). Songs live at /song/, so unlike
// the old single page this does not use the current path.
export function rootUris(): { withSlash: string; withoutSlash: string } {
  const origin = pageUrl().origin;
  return { withSlash: `${origin}/`, withoutSlash: origin };
}

export function effectiveRedirectUri(configured: string): string {
  if (configured.trim()) return configured.trim();
  const saved = lsGet(LS.redirectUri);
  if (saved && saved.trim()) return saved.trim();
  return rootUris().withSlash;
}

function generateRandomString(length: number): string {
  const possible = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~";
  const values = crypto.getRandomValues(new Uint8Array(length));
  return values.reduce((acc, x) => acc + possible[x % possible.length], "");
}

async function generateCodeChallenge(verifier: string): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(verifier));
  return btoa(String.fromCharCode(...new Uint8Array(digest)))
    .replace(/=/g, "")
    .replace(/\+/g, "-")
    .replace(/\//g, "_");
}

export async function startSpotifyAuth(clientId: string, redirectUri: string, pending: PendingPlaylist) {
  lsSet(LS.pending, JSON.stringify(pending));

  const verifier = generateRandomString(64);
  const challenge = await generateCodeChallenge(verifier);
  lsSet(LS.verifier, verifier);
  lsSet(LS.clientId, clientId);
  lsSet(LS.usedRedirectUri, redirectUri);

  const params = new URLSearchParams({
    client_id: clientId,
    response_type: "code",
    redirect_uri: redirectUri,
    scope: "playlist-modify-public playlist-modify-private",
    code_challenge_method: "S256",
    code_challenge: challenge,
  });
  const authUrl = `https://accounts.spotify.com/authorize?${params.toString()}`;
  // The Space shows the app in an iframe on huggingface.co; Spotify's login page refuses to be framed
  if (window.self !== window.top && window.open(authUrl, "_top")) return;
  window.open(authUrl, "_self");
}

export interface TokenResponse {
  access_token?: string;
  refresh_token?: string;
  error?: string;
  error_description?: string;
}

export async function exchangeCode(
  clientId: string, code: string, redirectUri: string, codeVerifier: string,
): Promise<TokenResponse> {
  const res = await fetch("https://accounts.spotify.com/api/token", {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({
      client_id: clientId,
      grant_type: "authorization_code",
      code,
      redirect_uri: redirectUri,
      code_verifier: codeVerifier,
    }),
  });
  return res.json();
}

// Spotify's own error text from a failed response, e.g. "403: User not registered in the Developer Dashboard"
async function spotifyError(res: Response): Promise<string> {
  let msg = res.statusText;
  try {
    const body = await res.json();
    if (body.error) msg = body.error.message || body.error_description || body.error;
  } catch {}
  return `${res.status}: ${msg}`;
}

export class SpotifyUnauthorized extends Error {}

export interface CreatedPlaylist {
  added: number;
  url?: string;
}

async function findTrackUri(token: string, t: PlaylistTrack): Promise<string | null> {
  try {
    const cleanTitle = t.title.replace(/\(.*?\)|\.?/g, "").trim();
    const cleanArtist = t.artist.replace(/feat\..*/i, "").trim();
    const queries = [`track:${cleanTitle} artist:${cleanArtist}`, `${cleanTitle} ${cleanArtist}`];
    for (const q of queries) {
      const sRes = await fetch(`https://api.spotify.com/v1/search?q=${encodeURIComponent(q)}&type=track&limit=1`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (sRes.ok) {
        const sData = await sRes.json();
        const items = sData.tracks ? sData.tracks.items : [];
        if (items.length > 0) return items[0].uri;
      }
    }
  } catch (e) {
    console.warn("Spotify search failed for", t.title, e);
  }
  return null;
}

export async function createSpotifyPlaylist(
  token: string, seedTitle: string, tracks: PlaylistTrack[],
): Promise<CreatedPlaylist> {
  // 1. Current Spotify user (also checks the token)
  const meRes = await fetch("https://api.spotify.com/v1/me", { headers: { Authorization: `Bearer ${token}` } });
  if (meRes.status === 401) throw new SpotifyUnauthorized("Session expired");
  if (!meRes.ok) throw new Error(await spotifyError(meRes));

  // 2. Create the playlist in the user's account (POST /users/{id}/playlists was removed for
  // development-mode apps in February 2026; /me/playlists replaces it)
  const createRes = await fetch("https://api.spotify.com/v1/me/playlists", {
    method: "POST",
    headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
    body: JSON.stringify({
      name: `Echo: ${seedTitle} Mix`,
      description: `Acoustic and emotional discovery mix generated by Echo AI from "${seedTitle}". Zero collaborative filtering.`,
      public: true,
    }),
  });
  if (!createRes.ok) throw new Error(await spotifyError(createRes));
  const playlist = await createRes.json();

  // 3. Spotify track URIs, searched concurrently
  const trackUris = (await Promise.all(tracks.map((t) => findTrackUri(token, t)))).filter(
    (u): u is string => Boolean(u));

  // 4. Add tracks (/tracks was renamed /items in February 2026)
  if (trackUris.length > 0) {
    const addRes = await fetch(`https://api.spotify.com/v1/playlists/${playlist.id}/items`, {
      method: "POST",
      headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
      body: JSON.stringify({ uris: trackUris }),
    });
    if (!addRes.ok) throw new Error("playlist made but songs not added, " + (await spotifyError(addRes)));
  }

  return { added: trackUris.length, url: playlist.external_urls?.spotify };
}

export function toPlaylistTracks(tracks: Track[]): PlaylistTrack[] {
  return tracks.map((t) => ({ title: t.title, artist: t.artist }));
}

"use client";

import { createContext, useCallback, useContext, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { getConfig, pageUrl } from "@/lib/api";
import {
  LS, type PendingPlaylist, type PlaylistTrack, SpotifyUnauthorized, createSpotifyPlaylist, effectiveRedirectUri,
  exchangeCode, lsGet, lsRemove, lsSet, rootUris, startSpotifyAuth,
} from "@/lib/spotify";
import { useToast } from "./Toast";

interface PlaylistSource {
  seed: string;
  tracks: PlaylistTrack[];
}

interface SpotifyValue {
  creating: boolean;
  // The songs on screen, which the button and the settings modal's login export
  setSource: (source: PlaylistSource | null) => void;
  createPlaylist: () => Promise<void>;
  openSettings: () => void;
}

const SpotifyContext = createContext<SpotifyValue>({
  creating: false,
  setSource: () => {},
  createPlaylist: async () => {},
  openSettings: () => {},
});

export function useSpotify() {
  return useContext(SpotifyContext);
}

// The code from Spotify's redirect is single-use; React's dev double-mount must not exchange it twice.
let callbackHandled = false;

export function SpotifyProvider({ children }: { children: React.ReactNode }) {
  const showToast = useToast();
  const router = useRouter();
  const config = useRef({ clientId: "", redirectUri: "" });
  const source = useRef<PlaylistSource | null>(null);
  const [creating, setCreating] = useState(false);
  const [modalOpen, setModalOpen] = useState(false);
  const [clientIdInput, setClientIdInput] = useState("");
  const [redirectInput, setRedirectInput] = useState("");
  const [uris, setUris] = useState({ withSlash: "", withoutSlash: "" });

  const openSettings = useCallback(() => {
    setClientIdInput(config.current.clientId || lsGet(LS.clientId) || "");
    setRedirectInput(lsGet(LS.redirectUri) || effectiveRedirectUri(config.current.redirectUri));
    setUris(rootUris());
    setModalOpen(true);
  }, []);

  const pendingFrom = useCallback((src: PlaylistSource | null): PendingPlaylist => {
    const here = pageUrl();
    const returnTo = here.pathname + here.search;
    if (src && src.tracks.length > 0) return { ...src, returnTo };
    // Nothing on screen (e.g. the modal opened after a failed login): keep what the failed login was exporting
    try {
      const prev = JSON.parse(lsGet(LS.pending) || "null") as PendingPlaylist | null;
      if (prev) return prev;
    } catch {}
    return { seed: src?.seed || "Echo Mix", tracks: [], returnTo };
  }, []);

  const execute = useCallback(async (token: string, seed: string, tracks: PlaylistTrack[]) => {
    setCreating(true);
    showToast("Creating Spotify playlist...");
    try {
      const { added, url } = await createSpotifyPlaylist(token, seed, tracks);
      showToast(`Playlist created in Spotify with ${added} of ${tracks.length} songs! Opening...`);
      if (url) window.open(url, "_blank");
    } catch (err) {
      if (err instanceof SpotifyUnauthorized) {
        lsRemove(LS.token);
        showToast("Session expired, reconnecting...");
        const cid = config.current.clientId || lsGet(LS.clientId);
        if (cid) {
          await startSpotifyAuth(cid, effectiveRedirectUri(config.current.redirectUri),
            { seed, tracks, returnTo: pageUrl().pathname + pageUrl().search });
        } else {
          openSettings();
        }
        return;
      }
      console.error("Create playlist error:", err);
      showToast("Error creating playlist: " + (err as Error).message);
    } finally {
      setCreating(false);
    }
  }, [showToast, openSettings]);

  useEffect(() => {
    getConfig().then((cfg) => {
      if (cfg.spotify_client_id) config.current.clientId = cfg.spotify_client_id;
      if (cfg.spotify_redirect_uri) config.current.redirectUri = cfg.spotify_redirect_uri;
    }).catch(() => {});
  }, []);

  // Spotify sends the browser back to the site root with ?code=... or ?error=...
  useEffect(() => {
    if (callbackHandled) return;
    const params = pageUrl().searchParams;
    const code = params.get("code");
    const authError = params.get("error");
    if (!code && !authError) return;
    callbackHandled = true;
    window.history.replaceState({}, document.title, pageUrl().pathname);

    if (authError) {
      showToast("Spotify Auth Error: " + authError);
      lsRemove(LS.confirmed);
      openSettings();
      return;
    }

    const codeVerifier = lsGet(LS.verifier);
    const clientId = config.current.clientId || lsGet(LS.clientId);
    const redirectUri = lsGet(LS.usedRedirectUri) || effectiveRedirectUri(config.current.redirectUri);
    if (!code || !codeVerifier || !clientId) return;

    (async () => {
      showToast("Connecting to Spotify...");
      try {
        const tokenData = await exchangeCode(clientId, code, redirectUri, codeVerifier);
        if (!tokenData.access_token) {
          showToast("Spotify Auth Failed: " + (tokenData.error_description || tokenData.error || "Unknown"));
          openSettings();
          return;
        }
        lsSet(LS.token, tokenData.access_token);
        if (tokenData.refresh_token) lsSet(LS.refreshToken, tokenData.refresh_token);
        const raw = lsGet(LS.pending);
        if (raw) {
          const pending = JSON.parse(raw) as PendingPlaylist;
          lsRemove(LS.pending);
          if (pending.returnTo && pending.returnTo !== "/") router.replace(pending.returnTo);
          showToast("Spotify Connected! Creating playlist...");
          await execute(tokenData.access_token, pending.seed, pending.tracks);
        }
      } catch (e) {
        console.error("Token exchange failed:", e);
        showToast("Spotify Token Exchange Error: " + (e as Error).message);
      }
    })();
  }, [showToast, openSettings, execute, router]);

  const setSource = useCallback((src: PlaylistSource | null) => {
    source.current = src;
  }, []);

  const createPlaylist = useCallback(async () => {
    const src = source.current;
    if (!src || src.tracks.length === 0) {
      showToast("Search or identify a song first!");
      return;
    }
    const token = lsGet(LS.token);
    if (token) {
      await execute(token, src.seed, src.tracks);
      return;
    }
    const savedClientId = config.current.clientId || lsGet(LS.clientId);
    const configuredUri = lsGet(LS.redirectUri) || effectiveRedirectUri(config.current.redirectUri);
    if (savedClientId && lsGet(LS.confirmed)) {
      await startSpotifyAuth(savedClientId, configuredUri, pendingFrom(src));
    } else {
      openSettings();
    }
  }, [showToast, execute, openSettings, pendingFrom]);

  const chooseUri = (uri: string, label: string) => {
    setRedirectInput(uri);
    lsSet(LS.redirectUri, uri);
    showToast(label);
  };

  const copyUri = () => {
    navigator.clipboard.writeText(redirectInput.trim())
      .then(() => showToast("📋 Redirect URI copied to clipboard!"))
      .catch(() => {
        (document.getElementById("spotifyRedirectUriInput") as HTMLInputElement | null)?.select();
        document.execCommand("copy");
        showToast("📋 Redirect URI copied!");
      });
  };

  const login = async () => {
    const clientId = clientIdInput.trim();
    const redirectUri = redirectInput.trim() || effectiveRedirectUri(config.current.redirectUri);
    if (!clientId) {
      alert("Please enter your Spotify Client ID");
      return;
    }
    lsSet(LS.clientId, clientId);
    lsSet(LS.redirectUri, redirectUri);
    lsSet(LS.confirmed, "true");
    setModalOpen(false);
    await startSpotifyAuth(clientId, redirectUri, pendingFrom(source.current));
  };

  return (
    <SpotifyContext.Provider value={{ creating, setSource, createPlaylist, openSettings }}>
      {children}
      {modalOpen && (
        <div className="modal-backdrop" onClick={(e) => e.target === e.currentTarget && setModalOpen(false)}>
          <div className="modal-card" role="dialog" aria-modal="true" aria-labelledby="spotifyModalTitle">
            <div className="modal-header">
              <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <SpotifyLogo size={24} fill="#1DB954" />
                <div>
                  <h3 id="spotifyModalTitle" style={{ fontSize: 16, fontWeight: 700 }}>Create Spotify Playlist</h3>
                  <p style={{ fontSize: 12, color: "var(--text-muted)" }}>
                    Authorize Echo to create this playlist directly in your Spotify account
                  </p>
                </div>
              </div>
              <button className="modal-close" title="Close" onClick={() => setModalOpen(false)}>&times;</button>
            </div>

            <div className="field">
              <label className="field-label" htmlFor="spotifyClientIdInput">Spotify Client ID</label>
              <input id="spotifyClientIdInput" className="field-input" type="text"
                placeholder="Paste your Spotify Client ID..." value={clientIdInput}
                onChange={(e) => setClientIdInput(e.target.value)} />
            </div>

            <div className="field" style={{ marginBottom: 16 }}>
              <div className="field-row">
                <label className="field-label" htmlFor="spotifyRedirectUriInput">Active Redirect URI</label>
                <div style={{ display: "flex", gap: 6 }}>
                  <button type="button" className="btn btn-secondary btn-tiny" title="Use URI ending with /"
                    onClick={() => chooseUri(uris.withSlash, "Set Redirect URI with /")}>With /</button>
                  <button type="button" className="btn btn-secondary btn-tiny" title="Use URI without trailing /"
                    onClick={() => chooseUri(uris.withoutSlash, "Set Redirect URI without /")}>Without /</button>
                </div>
              </div>
              <div style={{ display: "flex", gap: 8 }}>
                <input id="spotifyRedirectUriInput" className="field-input" style={{ flex: 1 }} type="text"
                  value={redirectInput} onChange={(e) => setRedirectInput(e.target.value)} />
                <button type="button" className="btn btn-secondary"
                  style={{ fontSize: 12, padding: "0 14px", whiteSpace: "nowrap" }} onClick={copyUri}>📋 Copy</button>
              </div>

              <div className="uri-help">
                <strong>⚠️ Fix &quot;redirect_uri: Not matching configuration&quot;:</strong>
                In <a href="https://developer.spotify.com/dashboard" target="_blank" rel="noreferrer">Spotify Developer
                Dashboard</a> ➔ your App ➔ <b>Settings</b> ➔ <b>Redirect URIs</b>:<br />
                Click <b>Add</b>, paste <b>both</b> of these URIs, and click <b>Save</b> at bottom:
                <ul>
                  <li>{uris.withSlash}</li>
                  <li>{uris.withoutSlash}</li>
                </ul>
              </div>
            </div>

            <button className="btn-spotify" style={{ width: "100%", justifyContent: "center", padding: 12 }}
              onClick={login}>
              <span>Log in with Spotify &amp; Create Playlist</span>
            </button>
          </div>
        </div>
      )}
    </SpotifyContext.Provider>
  );
}

export function SpotifyLogo({ size = 18, fill = "#000" }: { size?: number; fill?: string }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill={fill} aria-hidden="true">
      <path d="M12 0C5.4 0 0 5.4 0 12s5.4 12 12 12 12-5.4 12-12S18.6 0 12 0zm5.5 17.3c-.2.4-.7.5-1.1.3-3-1.8-6.8-2.2-11.3-1.2-.5.1-.9-.2-1-.7-.1-.5.2-.9.7-1 4.9-1.1 9.1-.6 12.4 1.5.5.2.6.7.3 1.1zm1.5-3.3c-.3.4-.8.6-1.3.3-3.5-2.1-8.7-2.7-12.8-1.5-.5.1-1-.2-1.2-.7-.1-.5.2-1 .7-1.2 4.7-1.4 10.5-.7 14.5 1.7.4.3.6.9.3 1.4zm.1-3.4c-4.1-2.5-10.9-2.7-14.9-1.5-.6.2-1.3-.2-1.5-.8-.2-.6.2-1.3.8-1.5 4.6-1.4 12-1.1 16.7 1.7.6.3.8 1.1.4 1.7-.3.5-1.1.7-1.6.4z" />
    </svg>
  );
}

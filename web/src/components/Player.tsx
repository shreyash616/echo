"use client";

import { createContext, useCallback, useContext, useEffect, useRef, useState } from "react";
import { useToast } from "./Toast";

// One <audio> for the whole app, so starting a preview stops the one playing, on any page.
interface PlayerValue {
  playingUrl: string | null;
  toggle: (url: string | null | undefined) => void;
}

const PlayerContext = createContext<PlayerValue>({ playingUrl: null, toggle: () => {} });

export function usePlayer() {
  return useContext(PlayerContext);
}

export function PlayerProvider({ children }: { children: React.ReactNode }) {
  const showToast = useToast();
  const audio = useRef<HTMLAudioElement>(null);
  const [playingUrl, setPlayingUrl] = useState<string | null>(null);

  const toggle = useCallback((url: string | null | undefined) => {
    const el = audio.current;
    if (!el) return;
    if (!url) {
      alert("No audio preview available for this track.");
      return;
    }
    if (el.src === url && !el.paused) {
      el.pause();
      setPlayingUrl(null);
    } else {
      el.src = url;
      el.play().catch((err: DOMException) => {
        // AbortError is just another preview starting; anything else (e.g. an expired preview link) is real
        if (err.name === "AbortError") return;
        setPlayingUrl(null);
        showToast("This preview could not be played");
      });
      setPlayingUrl(url);
    }
  }, [showToast]);

  useEffect(() => {
    const el = audio.current;
    if (!el) return;
    const ended = () => setPlayingUrl(null);
    el.addEventListener("ended", ended);
    return () => el.removeEventListener("ended", ended);
  }, []);

  return (
    <PlayerContext.Provider value={{ playingUrl, toggle }}>
      {children}
      <audio ref={audio} />
    </PlayerContext.Provider>
  );
}

export function PlayIcon({ size = 16 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
      <polygon points="5 3 19 12 5 21 5 3" />
    </svg>
  );
}

export function PauseIcon({ size = 16 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
      <rect x="6" y="4" width="4" height="16" />
      <rect x="14" y="4" width="4" height="16" />
    </svg>
  );
}

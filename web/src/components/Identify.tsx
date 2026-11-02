"use client";

import { createContext, useCallback, useContext, useState } from "react";
import { identify } from "@/lib/api";
import type { Track } from "@/lib/types";

// The result of a recording. It has no URL of its own (a mic clip may match no song), so it lives here and
// the home page shows it; recording from another page brings you home to it.
export interface IdentifyResult {
  hero: Track;
  recommendations: Track[];
}

interface IdentifyValue {
  busy: boolean;
  result: IdentifyResult | null;
  run: (blob: Blob) => Promise<void>;
  setRecommendations: (recs: Track[]) => void;
}

const IdentifyContext = createContext<IdentifyValue>({
  busy: false,
  result: null,
  run: async () => {},
  setRecommendations: () => {},
});

export function useIdentify() {
  return useContext(IdentifyContext);
}

export function IdentifyProvider({ children }: { children: React.ReactNode }) {
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<IdentifyResult | null>(null);

  const run = useCallback(async (blob: Blob) => {
    setBusy(true);
    setResult(null);
    try {
      const data = await identify(blob);
      let hero = data.identified || data.track || null;
      const recommendations = Array.isArray(data.recommendations) ? data.recommendations : [];
      if (!hero && recommendations.length > 0) {
        hero = {
          id: "",
          title: "Microphone Audio Analysis",
          artist: "Live Input",
          album: "Direct Spectral Embedding",
          energy: recommendations[0].energy ?? 0.5,
          valence: recommendations[0].valence ?? 0.5,
          vibes: recommendations[0].vibes || ["Acoustic Discovery"],
          lyricalIntent: "Intimate / Confessional",
        };
      }
      if (hero) setResult({ hero, recommendations });
    } catch (err) {
      alert("Audio recognition error: " + (err as Error).message);
    } finally {
      setBusy(false);
    }
  }, []);

  const setRecommendations = useCallback((recs: Track[]) => {
    setResult((r) => (r ? { ...r, recommendations: recs } : r));
  }, []);

  return (
    <IdentifyContext.Provider value={{ busy, result, run, setRecommendations }}>
      {children}
    </IdentifyContext.Provider>
  );
}

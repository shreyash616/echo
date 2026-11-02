"use client";

import { useSearchParams } from "next/navigation";
import { useEffect, useState } from "react";
import InputSection, { Loader } from "@/components/InputSection";
import SongRow from "@/components/SongRow";
import { FALLBACK_ART, remember, searchTracks } from "@/lib/api";
import type { Track } from "@/lib/types";

interface Result {
  name: string;
  list: Track[];
  error?: string;
}

// Songs credited to the artist, including features ("A, B" or "A feat. B")
function byArtist(tracks: Track[], name: string): Track[] {
  const n = name.toLowerCase();
  return tracks.filter((t) => (t.artist || "").toLowerCase().includes(n));
}

export default function ArtistView() {
  const name = (useSearchParams().get("name") || "").trim();
  const [result, setResult] = useState<Result | null>(null);

  useEffect(() => {
    if (!name) return;
    const ctl = new AbortController();
    searchTracks(name, 20, ctl.signal).then(
      (all) => {
        const list = byArtist(all, name);
        remember(list);
        setResult({ name, list });
      },
      (err) => {
        if (!ctl.signal.aborted) setResult({ name, list: [], error: (err as Error).message });
      });
    return () => ctl.abort();
  }, [name]);

  useEffect(() => {
    if (name) document.title = `${name} · echo`;
  }, [name]);

  const current = result && result.name === name ? result : null;

  return (
    <>
      <InputSection />
      {!name && <div className="empty-note">No artist given.</div>}
      {name && !current && <Loader text={`Finding songs by ${name}...`} />}
      {current && (
        <div className="recs-section">
          <div className="page-heading">
            <img src={current.list[0]?.albumArtUrl || `${FALLBACK_ART}?w=200`} alt="" />
            <div>
              <div className="hero-tagline">Artist</div>
              <h2>{name}</h2>
              <p>
                {current.list.length} {current.list.length === 1 ? "song" : "songs"} in search
                {" · "}
                <a href={`https://open.spotify.com/search/${encodeURIComponent(name)}`} target="_blank"
                  rel="noreferrer" style={{ color: "var(--teal)" }}>Open in Spotify</a>
              </p>
            </div>
          </div>
          <div className="recs-grid">
            {current.error && <div className="empty-note">Search error: {current.error}</div>}
            {!current.error && current.list.length === 0 && (
              <div className="empty-note">No songs by {name} found in search.</div>
            )}
            {current.list.map((t, i) => <SongRow key={`${t.id}-${i}`} track={t} rank={i} />)}
          </div>
        </div>
      )}
    </>
  );
}

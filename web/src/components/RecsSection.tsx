"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useMemo, useRef, useState } from "react";
import { FALLBACK_ART, artistHref, decadeOf, remember, songHref } from "@/lib/api";
import { toPlaylistTracks } from "@/lib/spotify";
import type { Track } from "@/lib/types";
import { THEME_STYLE, getVibeStyle, pillStyle } from "@/lib/vibes";
import { PauseIcon, PlayIcon, usePlayer } from "./Player";
import { SpotifyLogo, useSpotify } from "./Spotify";

function diversityLabel(val: number): string {
  if (val >= 0.85) return `Strict (${val.toFixed(2)})`;
  if (val >= 0.65) return `Balanced (${val.toFixed(2)})`;
  return `Discovery (${val.toFixed(2)})`;
}

const HIDDEN_REASONS = ["bpm", "camelot", "key", "vocal"];

interface Props {
  seedTitle: string;
  tracks: Track[];
  dim: boolean;
  diversity: number;
  onDiversity: (value: number) => void;
}

export default function RecsSection({ seedTitle, tracks, dim, diversity, onDiversity }: Props) {
  const spotify = useSpotify();
  // Sorted by matchScore so the ranks read top-down, as the old page did
  const sorted = useMemo(() => [...tracks].sort((a, b) => (b.matchScore ?? 0) - (a.matchScore ?? 0)), [tracks]);

  const { setSource } = spotify;
  useEffect(() => {
    setSource({ seed: seedTitle || "Echo Mix", tracks: toPlaylistTracks(sorted) });
    return () => setSource(null);
  }, [setSource, seedTitle, sorted]);

  return (
    <div className="recs-section">
      <div className="section-header">
        <div>
          <h3 className="section-title">Similar Acoustic Vibe</h3>
          <span className="badge">Neural + MMR Diversity Reranked</span>
        </div>
        <div className="header-controls">
          <button className="btn-spotify" title="Create this playlist on Spotify" disabled={spotify.creating}
            onClick={() => spotify.createPlaylist()}>
            <SpotifyLogo />
            <span>{spotify.creating ? "Creating in Spotify..." : "Create Spotify Playlist"}</span>
          </button>
          <button className="btn btn-secondary btn-settings" title="Spotify App Settings & Redirect URI"
            aria-label="Spotify settings" onClick={spotify.openSettings}>
            ⚙️
          </button>
          <DiversitySlider key={diversity} value={diversity} onCommit={onDiversity} />
        </div>
      </div>
      <div className={`recs-grid${dim ? " dim" : ""}`}>
        {sorted.length === 0
          ? <div className="empty-note">No similar tracks found for this song.</div>
          : sorted.map((t, i) => <RecCard key={`${t.id}-${i}`} track={t} rank={i} diversity={diversity} />)}
      </div>
    </div>
  );
}

function DiversitySlider({ value, onCommit }: { value: number; onCommit: (v: number) => void }) {
  const [val, setVal] = useState(value);
  const input = useRef<HTMLInputElement>(null);

  // The label follows the drag (React's onChange is the input event); the reload waits for the native change
  // event, which fires on release, plus 150 ms as on the old page
  useEffect(() => {
    const el = input.current;
    if (!el) return;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const onChange = () => {
      clearTimeout(timer);
      timer = setTimeout(() => onCommit(parseFloat(el.value)), 150);
    };
    el.addEventListener("change", onChange);
    return () => {
      clearTimeout(timer);
      el.removeEventListener("change", onChange);
    };
  }, [onCommit]);

  return (
    <div className="discovery-control" title="Tune between strict acoustic clones and broader sonic discovery">
      <span className="discovery-label">Discovery Mode</span>
      <input ref={input} type="range" className="discovery-slider" min="0.50" max="0.95" step="0.05" value={val}
        aria-label="Discovery mode" onChange={(e) => setVal(parseFloat(e.target.value))} />
      <span className="discovery-val">{diversityLabel(val)}</span>
    </div>
  );
}

function RecCard({ track, rank, diversity }: { track: Track; rank: number; diversity: number }) {
  const router = useRouter();
  const { playingUrl, toggle } = usePlayer();
  const playing = !!track.previewUrl && playingUrl === track.previewUrl;
  const href = songHref(track, diversity);
  const matchPct = track.matchScore !== undefined && track.matchScore !== null
    ? Math.round(track.matchScore * 100)
    : Math.max(70, 96 - rank * 2);
  const instSnippet = (track.instruments || []).slice(0, 2).join(" • ");
  const decade = decadeOf(track);
  const crossLanguage = track.languageAffinity !== null && track.languageAffinity !== undefined
    && track.languageAffinity < 0.99;
  const reasons = (track.matchReasons || [])
    .filter((r) => !HIDDEN_REASONS.some((h) => r.toLowerCase().includes(h)));
  const open = () => {
    remember([track]);
    router.push(href);
  };

  return (
    // Clicking a card makes it the hero: it opens that song's page
    <div className="rec-card" onClick={open}>
      <div className="rec-rank">#{rank + 1}</div>
      <img className="rec-art" src={track.albumArtUrl || `${FALLBACK_ART}?w=120`} alt="" />
      <div className="rec-details">
        <div className="rec-title">
          <Link href={href} onClick={(e) => { e.stopPropagation(); remember([track]); }}>{track.title}</Link>
        </div>
        <div className="rec-artist">
          <Link className="artist-link" href={artistHref(track.artist)} onClick={(e) => e.stopPropagation()}>
            {track.artist}
          </Link>
        </div>
        <div className="rec-structural-tags">
          {(track.vibes || []).slice(0, 1).map((v) => (
            <span key={v} className="rec-tag" style={pillStyle(getVibeStyle(v))}>🌊 {v}</span>
          ))}
          {track.theme && <span className="rec-tag" style={pillStyle(THEME_STYLE)}>📖 {track.theme}</span>}
          {crossLanguage && (
            <span className="rec-tag"
              style={{ background: "rgba(45,212,191,0.15)", borderColor: "rgba(45,212,191,0.3)", color: "#2dd4bf" }}>
              🌐 Feel Cross-Language
            </span>
          )}
          {decade && <span className="rec-tag">📅 ({decade})</span>}
          {instSnippet && <span className="rec-tag">🎸 {instSnippet}</span>}
        </div>
        {reasons.length > 0 && (
          <div className="rec-reasons">
            {reasons.map((r) => <span key={r} className="reason-badge">✨ {r}</span>)}
          </div>
        )}
      </div>
      <div className="rec-score" title={`${matchPct}% match`}>{matchPct}%</div>
      <button className="rec-play-btn" title={playing ? "Pause preview" : "Play preview"}
        aria-label={playing ? "Pause preview" : "Play preview"}
        onClick={(e) => { e.stopPropagation(); toggle(track.previewUrl); }}>
        {playing ? <PauseIcon size={14} /> : <PlayIcon size={14} />}
      </button>
    </div>
  );
}

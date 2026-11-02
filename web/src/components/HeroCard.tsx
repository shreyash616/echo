"use client";

import Link from "next/link";
import { forwardRef } from "react";
import { FALLBACK_ART, artistHref, decadeOf, songHref } from "@/lib/api";
import type { Track } from "@/lib/types";
import { THEME_STYLE, getVibeStyle, pillStyle } from "@/lib/vibes";
import { PauseIcon, PlayIcon, usePlayer } from "./Player";

interface Props {
  hero: Track;
  // Link the title to the song's own page (results of a recording are not on one)
  linkToSong?: boolean;
  actions?: React.ReactNode;
}

const HeroCard = forwardRef<HTMLDivElement, Props>(function HeroCard({ hero, linkToSong, actions }, ref) {
  const { playingUrl, toggle } = usePlayer();
  const playing = !!hero.previewUrl && playingUrl === hero.previewUrl;
  const decade = decadeOf(hero);
  const instruments = hero.instruments && hero.instruments.length ? hero.instruments : ["Harmonic & Rhythmic Body"];
  const title = hero.title || "Unknown Title";
  const artist = hero.artist || "Unknown Artist";

  return (
    <div className="hero-card" ref={ref}>
      <div className="hero-content">
        <img className="hero-art" src={hero.albumArtUrl || `${FALLBACK_ART}?w=300`} alt="Album Art" />
        <div className="hero-info">
          <div className="hero-tagline">Identified Acoustic Source</div>
          <h2 className="hero-title">{linkToSong && hero.id ? <Link href={songHref(hero)}>{title}</Link> : title}</h2>
          <div className="hero-artist">
            {hero.artist ? <Link className="artist-link" href={artistHref(hero.artist)}>{artist}</Link> : artist}
          </div>

          <div className="vibe-pills" style={{ marginBottom: 12 }}>
            {(hero.vibes || []).map((v) => (
              <span key={v} className="vibe-pill" style={pillStyle(getVibeStyle(v))}>🌊 {v}</span>
            ))}
            {hero.theme && <span className="vibe-pill" style={pillStyle(THEME_STYLE)}>📖 {hero.theme}</span>}
          </div>

          <div className="structural-breakdown">
            <div className="struct-row">
              <span className="struct-label">🎸 Instruments</span>
              <div className="struct-badges">
                {instruments.map((inst) => <span key={inst} className="struct-badge">{inst}</span>)}
              </div>
            </div>
            {hero.spatialStaging && (
              <div className="struct-row">
                <span className="struct-label">🎧 Soundstage</span>
                <span className="struct-val">{hero.spatialStaging}</span>
              </div>
            )}
            {hero.lyricalIntent && (
              <div className="struct-row">
                <span className="struct-label">❤️ Emotion</span>
                <span className="struct-val">{hero.lyricalIntent}</span>
              </div>
            )}
          </div>

          <div className="hero-stats">
            {!!hero.felt_bpm && <span className="stat-pill">🎵 {hero.felt_bpm} BPM</span>}
            {hero.energy !== undefined && hero.energy !== null && (
              <span className="stat-pill">⚡ {Math.round(hero.energy * 100)}% Energy</span>
            )}
            {decade && <span className="stat-pill" style={{ color: "var(--teal)", fontWeight: 700 }}>📅 ({decade})</span>}
          </div>

          <div className="hero-actions">
            <button className="btn btn-play" onClick={() => toggle(hero.previewUrl)}>
              {playing ? <PauseIcon /> : <PlayIcon />}
              <span>{playing ? "Pause" : "Play Preview"}</span>
            </button>
            <a className="btn btn-secondary" target="_blank" rel="noreferrer"
              href={`https://open.spotify.com/search/${encodeURIComponent(hero.title + " " + hero.artist)}`}>
              Open in Spotify
            </a>
            {actions}
          </div>
        </div>
      </div>
    </div>
  );
});

export default HeroCard;

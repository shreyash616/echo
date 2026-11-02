"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { FALLBACK_ART, artistHref, remember, songHref } from "@/lib/api";
import type { Track } from "@/lib/types";

// A song in a search or artist list; opening it shows the song with its matches.
export default function SongRow({ track, rank }: { track: Track; rank: number }) {
  const router = useRouter();
  const href = songHref(track);
  const open = () => {
    remember([track]);
    router.push(href);
  };
  return (
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
          {" • "}{track.year ? `${track.year} • ` : ""}{track.album || "Single"}
        </div>
      </div>
      <div className="song-row-action">Select &amp; Match →</div>
    </div>
  );
}

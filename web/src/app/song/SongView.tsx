"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";
import HeroCard from "@/components/HeroCard";
import InputSection, { Loader } from "@/components/InputSection";
import RecsSection from "@/components/RecsSection";
import { useToast } from "@/components/Toast";
import { DEFAULT_DIVERSITY, HttpError, getRecommendations, getTrack, recall, remember } from "@/lib/api";
import type { Track } from "@/lib/types";

function parseDiversity(raw: string | null): number {
  const v = raw === null ? NaN : parseFloat(raw);
  return Number.isFinite(v) && v >= 0.5 && v <= 0.95 ? v : DEFAULT_DIVERSITY;
}

// Each result remembers the request it answers, so "loading" is simply "no result for the current URL yet".
interface TrackResult {
  req: string;
  hero?: Track;
  error?: string;
}

interface RecsResult {
  req: string;
  id: string;
  list: Track[];
}

export default function SongView() {
  const params = useSearchParams();
  const router = useRouter();
  const showToast = useToast();
  const id = params.get("id") || "";
  const key = params.get("key") || "";
  const diversity = parseDiversity(params.get("d"));
  const trackReq = `${id}|${key}`;
  const recsReq = `${id}|${key}|${diversity}`;

  const [track, setTrack] = useState<TrackResult | null>(null);
  const [recs, setRecs] = useState<RecsResult | null>(null);
  const heroRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!id) return;
    const ctl = new AbortController();
    getTrack(id, key, ctl.signal).then(
      (hero) => setTrack({ req: trackReq, hero }),
      (err) => {
        if (ctl.signal.aborted) return;
        // As on the old page, a failed lookup falls back to the song as the list showed it
        const seen = recall(id);
        const notFound = err instanceof HttpError && err.status === 404;
        setTrack({ req: trackReq, hero: seen, error: notFound ? "Song not found." : (err as Error).message });
      });
    return () => ctl.abort();
  }, [id, key, trackReq]);

  useEffect(() => {
    if (!id) return;
    const ctl = new AbortController();
    getRecommendations(id, key, diversity, ctl.signal).then(
      (list) => {
        remember(list);
        setRecs({ req: recsReq, id, list });
      },
      (err) => {
        if (ctl.signal.aborted) return;
        console.error("recommendations error:", err);
        // A failed re-rank keeps the cards on screen; a failed first load shows none
        setRecs((prev) => (prev && prev.id === id ? { ...prev, req: recsReq } : { req: recsReq, id, list: [] }));
      });
    return () => ctl.abort();
  }, [id, key, diversity, recsReq]);

  const trackLoaded = track?.req === trackReq;
  const hero = trackLoaded ? track.hero : undefined;
  const recsForSong = recs && recs.id === id ? recs : null;
  const ready = trackLoaded && !!hero && !!recsForSong;

  useEffect(() => {
    if (!hero) return;
    document.title = `${hero.title} — ${hero.artist} · echo`;
  }, [hero]);

  useEffect(() => {
    if (ready) heroRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [ready, id]);

  const onDiversity = useCallback((value: number) => {
    const next = new URLSearchParams(params.toString());
    if (value === DEFAULT_DIVERSITY) next.delete("d");
    else next.set("d", value.toFixed(2));
    router.replace(`/song/?${next}`, { scroll: false });
  }, [params, router]);

  const copyLink = () => {
    navigator.clipboard.writeText(window.location.href)
      .then(() => showToast("🔗 Song link copied!"))
      .catch(() => showToast("Copy failed"));
  };

  const seen = recall(id);
  const initialQuery = seen ? `${seen.title} - ${seen.artist}` : undefined;

  if (!id) {
    return (
      <>
        <InputSection />
        <div className="empty-note">No song picked. <Link href="/">Search for one</Link>.</div>
      </>
    );
  }

  return (
    <>
      <InputSection initialQuery={initialQuery} />
      {!ready && trackLoaded && !hero && (
        <div className="empty-note">{track.error || "Song not found."} <Link href="/">Search again</Link>.</div>
      )}
      {!ready && !(trackLoaded && !hero) && (
        <Loader text={seen
          ? `Analyzing acoustic dimensions & matching candidates for "${seen.title}"...`
          : "Analyzing acoustic dimensions & matching candidates..."} />
      )}
      {ready && hero && recsForSong && (
        <>
          <HeroCard ref={heroRef} hero={hero} actions={
            <button className="btn btn-secondary" onClick={copyLink} title="Copy a link to this song">🔗 Share</button>
          } />
          <RecsSection seedTitle={hero.title} tracks={recsForSong.list} dim={recsForSong.req !== recsReq}
            diversity={diversity} onDiversity={onDiversity} />
        </>
      )}
    </>
  );
}

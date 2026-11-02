"use client";

import { useSearchParams } from "next/navigation";
import { useEffect, useState } from "react";
import InputSection, { Loader } from "@/components/InputSection";
import SongRow from "@/components/SongRow";
import { remember, searchTracks } from "@/lib/api";
import type { Track } from "@/lib/types";

interface Result {
  q: string;
  list: Track[];
  error?: string;
}

export default function SearchView() {
  const q = (useSearchParams().get("q") || "").trim();
  const [result, setResult] = useState<Result | null>(null);

  useEffect(() => {
    if (!q) return;
    const ctl = new AbortController();
    searchTracks(q, 20, ctl.signal).then(
      (list) => {
        remember(list);
        setResult({ q, list });
      },
      (err) => {
        if (!ctl.signal.aborted) setResult({ q, list: [], error: (err as Error).message });
      });
    return () => ctl.abort();
  }, [q]);

  useEffect(() => {
    document.title = q ? `“${q}” — echo search` : "echo — Acoustic AI Music Discovery";
  }, [q]);

  const current = result && result.q === q ? result : null;

  return (
    <>
      <InputSection initialQuery={q} />
      {!q && <div className="empty-note">Type a song or artist above.</div>}
      {q && !current && <Loader text="Searching songs..." />}
      {current && (
        <div className="recs-section">
          <div className="section-header">
            <h3 className="section-title">Songs matching &ldquo;{q}&rdquo;</h3>
          </div>
          <div className="recs-grid">
            {current.error && <div className="empty-note">Search error: {current.error}</div>}
            {!current.error && current.list.length === 0 && <div className="empty-note">No matching songs found</div>}
            {current.list.map((t, i) => <SongRow key={`${t.id}-${i}`} track={t} rank={i} />)}
          </div>
        </div>
      )}
    </>
  );
}

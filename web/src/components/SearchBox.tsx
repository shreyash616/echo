"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { FALLBACK_ART, remember, searchHref, searchTracks, songHref } from "@/lib/api";
import type { Track } from "@/lib/types";

type Dropdown =
  | { state: "closed" }
  | { state: "hint"; text: string }
  | { state: "results"; matches: Track[] };

// Type-ahead: 250 ms after the last keystroke, top 5 songs. Enter opens the full results page.
export default function SearchBox({ initialQuery = "" }: { initialQuery?: string }) {
  const router = useRouter();
  const [query, setQuery] = useState(initialQuery);
  const [dropdown, setDropdown] = useState<Dropdown>({ state: "closed" });
  const box = useRef<HTMLDivElement>(null);
  const debounce = useRef<ReturnType<typeof setTimeout> | null>(null);
  const latest = useRef(0);

  const fetchMatches = async (q: string) => {
    const n = ++latest.current;
    setDropdown({ state: "hint", text: "Searching songs..." });
    try {
      const matches = await searchTracks(q, 5);
      if (n !== latest.current) return;
      remember(matches);
      setDropdown(matches.length === 0
        ? { state: "hint", text: "No matching songs found" }
        : { state: "results", matches });
    } catch (err) {
      if (n !== latest.current) return;
      setDropdown({ state: "hint", text: `Search error: ${(err as Error).message}` });
    }
  };

  const onChange = (value: string) => {
    setQuery(value);
    const q = value.trim();
    if (debounce.current) clearTimeout(debounce.current);
    if (q.length < 2) {
      latest.current++;
      setDropdown({ state: "closed" });
      return;
    }
    debounce.current = setTimeout(() => fetchMatches(q), 250);
  };

  useEffect(() => {
    const onDocClick = (e: MouseEvent) => {
      if (box.current && !box.current.contains(e.target as Node)) setDropdown({ state: "closed" });
    };
    document.addEventListener("click", onDocClick);
    return () => {
      document.removeEventListener("click", onDocClick);
      if (debounce.current) clearTimeout(debounce.current);
    };
  }, []);

  const select = (song: Track) => {
    latest.current++;
    setDropdown({ state: "closed" });
    setQuery(`${song.title} - ${song.artist}`);
    router.push(songHref(song));
  };

  return (
    <div className="search-box" ref={box}>
      <svg className="search-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor"
        strokeWidth="2" aria-hidden="true">
        <circle cx="11" cy="11" r="8" />
        <line x1="21" y1="21" x2="16.65" y2="16.65" />
      </svg>
      <input
        type="text"
        className="search-input"
        placeholder="Search any song or artist (e.g. Starboy, Drake)..."
        autoComplete="off"
        aria-label="Search songs"
        value={query}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Escape") {
            setDropdown({ state: "closed" });
          } else if (e.key === "Enter" && query.trim()) {
            if (debounce.current) clearTimeout(debounce.current);
            latest.current++;
            setDropdown({ state: "closed" });
            router.push(searchHref(query.trim()));
          }
        }}
      />
      {dropdown.state !== "closed" && (
        <div className="search-results-dropdown">
          {dropdown.state === "hint" && <div className="search-loading-hint">{dropdown.text}</div>}
          {dropdown.state === "results" && (
            <>
              {dropdown.matches.map((song) => (
                <button key={song.id} type="button" className="search-result-item" onClick={() => select(song)}>
                  <img className="search-result-thumb" src={song.albumArtUrl || `${FALLBACK_ART}?w=100`} alt="" />
                  <div className="search-result-info">
                    <div className="search-result-title">{song.title}</div>
                    <div className="search-result-artist">
                      {song.artist} • {song.year ? `${song.year} • ` : ""}{song.album || "Single"}
                    </div>
                  </div>
                  <div className="search-result-action">Select &amp; Match →</div>
                </button>
              ))}
              <Link className="search-see-all" href={searchHref(query.trim())}
                onClick={() => setDropdown({ state: "closed" })}>
                All results for &ldquo;{query.trim()}&rdquo; →
              </Link>
            </>
          )}
        </div>
      )}
    </div>
  );
}

"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import HeroCard from "@/components/HeroCard";
import { useIdentify } from "@/components/Identify";
import InputSection, { Loader } from "@/components/InputSection";
import RecsSection from "@/components/RecsSection";
import { DEFAULT_DIVERSITY, MIC_RECORDING_ID, getRecommendations, trackKey } from "@/lib/api";

// Home: search, record or hum. A recording's result shows here; a searched song opens /song/?id=...
export default function Home() {
  const { busy, result, setRecommendations } = useIdentify();
  const [diversity, setDiversity] = useState(DEFAULT_DIVERSITY);
  const [reloading, setReloading] = useState(false);
  const heroRef = useRef<HTMLDivElement>(null);
  const hero = result?.hero;

  useEffect(() => {
    if (hero) heroRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [hero]);

  const onDiversity = useCallback(async (value: number) => {
    setDiversity(value);
    // A clip that matched no song has no id to re-rank from
    if (!hero || !hero.id || hero.id === MIC_RECORDING_ID) return;
    setReloading(true);
    try {
      setRecommendations(await getRecommendations(hero.id, trackKey(hero), value));
    } catch (err) {
      console.error("reload recommendations error:", err);
    } finally {
      setReloading(false);
    }
  }, [hero, setRecommendations]);

  return (
    <>
      <InputSection />
      {busy && <Loader text="Fingerprinting & neural encoding..." />}
      {!busy && result && (
        <>
          <HeroCard ref={heroRef} hero={result.hero} linkToSong={result.hero.id !== MIC_RECORDING_ID} />
          <RecsSection seedTitle={result.hero.title} tracks={result.recommendations} dim={reloading}
            diversity={diversity} onDiversity={onDiversity} />
        </>
      )}
    </>
  );
}

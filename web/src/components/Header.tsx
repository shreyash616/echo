"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

export default function Header() {
  const viz = usePathname().startsWith("/visualizations");
  return (
    <header>
      <Link href="/" className="brand" style={{ textDecoration: "none" }}>
        <h1>echo</h1>
        <span className="badge">Acoustic AI</span>
      </Link>
      <p className="subtitle">Self-supervised music discovery powered by raw acoustic neural representations</p>

      <nav className="view-switch">
        <Link href="/" className={`btn ${viz ? "btn-secondary" : "btn-play"}`}>Music Discovery</Link>
        <Link href="/visualizations/" className={`btn ${viz ? "btn-play" : "btn-secondary"}`}>Model Visualizations</Link>
      </nav>
    </header>
  );
}

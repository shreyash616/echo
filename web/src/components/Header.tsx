import Link from "next/link";

export default function Header() {
  return (
    <header>
      <Link href="/" className="brand" style={{ textDecoration: "none" }}>
        <h1>echo</h1>
        <span className="badge">Acoustic AI</span>
      </Link>
      <p className="subtitle">Self-supervised music discovery powered by raw acoustic neural representations</p>
    </header>
  );
}

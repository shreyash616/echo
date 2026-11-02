import type { NextConfig } from "next";
import { PHASE_DEVELOPMENT_SERVER } from "next/constants";

// The build is a static export that FastAPI serves from the Space (backend/main.py mounts it at /), so the
// app calls /api and /static on its own origin. `next dev` has no FastAPI behind it, so it proxies those two
// paths to ECHO_BACKEND, the live Space by default. Rewrites are dev-only: a static export cannot have them.
const BACKEND = process.env.ECHO_BACKEND || "https://shreyash616-echo-backend.hf.space";

export default function config(phase: string): NextConfig {
  if (phase === PHASE_DEVELOPMENT_SERVER) {
    return {
      trailingSlash: true,
      async rewrites() {
        return [
          { source: "/api/:path*", destination: `${BACKEND}/api/:path*` },
          { source: "/static/:path*", destination: `${BACKEND}/static/:path*` },
        ];
      },
    };
  }
  return {
    output: "export",
    // /song/index.html rather than /song.html: Starlette's StaticFiles(html=True) serves the former for /song/
    trailingSlash: true,
  };
}

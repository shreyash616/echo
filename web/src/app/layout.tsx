import type { Metadata } from "next";
import { JetBrains_Mono, Plus_Jakarta_Sans } from "next/font/google";
import Header from "@/components/Header";
import { IdentifyProvider } from "@/components/Identify";
import { PlayerProvider } from "@/components/Player";
import { SpotifyProvider } from "@/components/Spotify";
import { ToastProvider } from "@/components/Toast";
import "./globals.css";

const sans = Plus_Jakarta_Sans({
  variable: "--font-sans",
  subsets: ["latin"],
  weight: ["300", "400", "500", "600", "700", "800"],
});

const mono = JetBrains_Mono({
  variable: "--font-mono",
  subsets: ["latin"],
  weight: ["400", "500"],
});

export const metadata: Metadata = {
  title: "echo — Acoustic AI Music Discovery",
  description: "Self-supervised music discovery powered by raw acoustic neural representations",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${sans.variable} ${mono.variable}`}>
      <body>
        <ToastProvider>
          <PlayerProvider>
            <IdentifyProvider>
              <SpotifyProvider>
                <div className="ambient-glow" />
                <div className="container">
                  <Header />
                  {children}
                </div>
              </SpotifyProvider>
            </IdentifyProvider>
          </PlayerProvider>
        </ToastProvider>
      </body>
    </html>
  );
}

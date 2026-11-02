export interface VibeStyle {
  bg: string;
  border: string;
  color: string;
}

const VIBE_COLOR_MAP: Record<string, VibeStyle> = {
  slow:        { bg: "rgba(139, 92, 246, 0.15)", border: "rgba(139, 92, 246, 0.35)", color: "#c4b5fd" },
  heavy:       { bg: "rgba(139, 92, 246, 0.15)", border: "rgba(139, 92, 246, 0.35)", color: "#c4b5fd" },
  sparse:      { bg: "rgba(45, 212, 191, 0.15)", border: "rgba(45, 212, 191, 0.35)", color: "#2dd4bf" },
  intimate:    { bg: "rgba(45, 212, 191, 0.15)", border: "rgba(45, 212, 191, 0.35)", color: "#2dd4bf" },
  hushed:      { bg: "rgba(148, 163, 184, 0.15)", border: "rgba(148, 163, 184, 0.35)", color: "#cbd5e1" },
  laid:        { bg: "rgba(251, 191, 36, 0.15)", border: "rgba(251, 191, 36, 0.35)", color: "#fcd34d" },
  raw:         { bg: "rgba(251, 146, 60, 0.15)", border: "rgba(251, 146, 60, 0.35)", color: "#fb923c" },
  psychedelic: { bg: "rgba(232, 121, 249, 0.15)", border: "rgba(232, 121, 249, 0.35)", color: "#e879f9" },
  punk:        { bg: "rgba(239, 68, 68, 0.15)", border: "rgba(239, 68, 68, 0.35)", color: "#fca5a5" },
  lean:        { bg: "rgba(196, 181, 253, 0.15)", border: "rgba(196, 181, 253, 0.35)", color: "#c4b5fd" },
  dark:        { bg: "rgba(99, 102, 241, 0.15)", border: "rgba(99, 102, 241, 0.35)", color: "#a5b4fc" },
  bright:      { bg: "rgba(251, 191, 36, 0.15)", border: "rgba(251, 191, 36, 0.35)", color: "#fcd34d" },
  folk:        { bg: "rgba(217, 119, 6, 0.15)", border: "rgba(217, 119, 6, 0.35)", color: "#fbbf24" },
  dance:       { bg: "rgba(168, 85, 247, 0.15)", border: "rgba(168, 85, 247, 0.35)", color: "#d8b4fe" },
  reggaeton:   { bg: "rgba(234, 88, 12, 0.15)", border: "rgba(234, 88, 12, 0.35)", color: "#fb923c" },
  reggae:      { bg: "rgba(34, 197, 94, 0.15)", border: "rgba(34, 197, 94, 0.35)", color: "#4ade80" },
  warm:        { bg: "rgba(251, 146, 60, 0.15)", border: "rgba(251, 146, 60, 0.35)", color: "#fdba74" },
  driving:     { bg: "rgba(239, 68, 68, 0.15)", border: "rgba(239, 68, 68, 0.35)", color: "#fca5a5" },
};

export const THEME_STYLE: VibeStyle = { bg: "rgba(245,158,11,0.15)", border: "rgba(245,158,11,0.35)", color: "#fbbf24" };

export function getVibeStyle(vibe: string | undefined): VibeStyle {
  if (!vibe) return { bg: "rgba(255,255,255,0.08)", border: "rgba(255,255,255,0.15)", color: "#94a3b8" };
  const firstWord = String(vibe).trim().split(/[\s-]+/)[0].toLowerCase();
  return VIBE_COLOR_MAP[firstWord] || { bg: "rgba(139, 92, 246, 0.15)", border: "rgba(139, 92, 246, 0.35)", color: "#c4b5fd" };
}

export function pillStyle(s: VibeStyle): React.CSSProperties {
  return { background: s.bg, borderColor: s.border, color: s.color };
}

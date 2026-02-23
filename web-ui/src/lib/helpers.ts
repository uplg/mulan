import type { SongInfo } from "./types";

const styleColors: Record<string, [string, string]> = {
  electronic: ["#00d4ff", "#0066ff"],
  ambient: ["#2d5a27", "#1a3a1a"],
  rock: ["#ff4444", "#aa2222"],
  pop: ["#ff69b4", "#ff1493"],
  jazz: ["#9b59b6", "#8e44ad"],
  classical: ["#f5e6d3", "#c9a86c"],
  "hip hop": ["#ffd700", "#ff8c00"],
  rap: ["#ff6600", "#cc3300"],
  metal: ["#333333", "#1a1a1a"],
  acoustic: ["#8b4513", "#654321"],
  piano: ["#2c3e50", "#1a252f"],
  instrumental: ["#1a1a2e", "#16213e"],
};

export function getGradientFromStyles(styles: string): string {
  const lower = styles.toLowerCase();
  for (const [style, colors] of Object.entries(styleColors)) {
    if (lower.includes(style)) {
      return `linear-gradient(135deg, ${colors[0]} 0%, ${colors[1]} 100%)`;
    }
  }
  return "linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%)";
}

export function getAlbumArtStyle(song: SongInfo | null): React.CSSProperties {
  return { background: getGradientFromStyles(song?.styles ?? "") };
}

export function formatDuration(seconds: number): string {
  if (!seconds || isNaN(seconds)) return "0:00";
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, "0")}`;
}

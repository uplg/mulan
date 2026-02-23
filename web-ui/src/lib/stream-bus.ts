/**
 * Imperative streaming segment channel.
 *
 * Segments and playback time live here — outside React state — so that
 * pushing a new segment does NOT trigger a React re-render of the entire
 * tree.  PlayerBar subscribes to notifications via a plain callback and
 * reads segments directly from the arrays stored here.
 */

export interface StreamBus {
  /** Accumulated blob URLs for each decoded audio segment. */
  segments: string[];
  /** Current playback position (seconds), written by PlayerBar. */
  time: number;
  /** Push a new segment URL and notify the listener. */
  push(url: string): void;
  /** Clear all segments and reset time. */
  clear(): void;
  /**
   * Callback invoked (synchronously) whenever a segment is pushed.
   * PlayerBar sets this once; CreateView never touches it.
   */
  onSegment: (() => void) | null;
}

/** Singleton shared by CreateView (producer) and PlayerBar (consumer). */
export const streamBus: StreamBus = {
  segments: [],
  time: 0,
  onSegment: null,

  push(url: string) {
    this.segments.push(url);
    this.onSegment?.();
  },

  clear() {
    // Revoke any lingering segment URLs would be caller's job (CreateView
    // already handles that).  Only reset transport state; do NOT touch
    // onSegment — the callback is owned by PlayerBar and registered once.
    this.segments = [];
    this.time = 0;
  },
};

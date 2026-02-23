import { useRef, useState, useEffect, useCallback } from "react";
import {
  Play,
  Pause,
  SkipBack,
  SkipForward,
  Shuffle,
  Repeat,
  Volume2,
  VolumeX,
  Download,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { useApp } from "@/lib/store";
import { api } from "@/lib/api";
import { streamBus } from "@/lib/stream-bus";
import { getAlbumArtStyle, formatDuration } from "@/lib/helpers";
import { cn } from "@/lib/utils";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Stop an audio element completely: pause, reset src, reset time. */
function killAudio(el: HTMLAudioElement | null) {
  if (!el) return;
  el.pause();
  el.removeAttribute("src");
  el.currentTime = 0;
}

// ---------------------------------------------------------------------------
// PlayerBar
// ---------------------------------------------------------------------------

export function PlayerBar() {
  const { state, dispatch } = useApp();

  // ---- Two <audio> elements for gapless double-buffering ----
  const audioARef = useRef<HTMLAudioElement>(null);
  const audioBRef = useRef<HTMLAudioElement>(null);

  /** Which element is currently the "active" (playing) one: "A" | "B". */
  const activeRef = useRef<"A" | "B">("A");

  const getActive = useCallback(
    () => (activeRef.current === "A" ? audioARef.current : audioBRef.current),
    [],
  );
  const getStandby = useCallback(
    () => (activeRef.current === "A" ? audioBRef.current : audioARef.current),
    [],
  );

  // ---- UI state ----
  const [currentTime, setCurrentTime] = useState(0);
  const [totalDuration, setTotalDuration] = useState(0);
  const [volume, setVolume] = useState(80);
  const [isMuted, setIsMuted] = useState(false);
  const [isShuffled, setIsShuffled] = useState(false);
  const [repeatMode, setRepeatMode] = useState<"none" | "all" | "one">("none");

  // ---- Streaming bookkeeping (refs, not state — no re-renders) ----
  /** Index into streamBus.segments of the segment currently playing. -1 = not streaming. */
  const streamIndexRef = useRef(-1);
  /** Index of the segment already preloaded on the standby element. */
  const preloadedIndexRef = useRef(-1);
  /** Cumulative seconds of all segments that finished before the current one. */
  const streamBaseTimeRef = useRef(0);
  /**
   * True when the active segment ended but the next segment hasn't arrived
   * from the server yet.  The onSegment callback checks this and resumes
   * playback as soon as the missing segment is pushed.
   */
  const waitingForSegmentRef = useRef(false);
  /** Pending seek position — consumed once after the next loadedmetadata event. */
  const pendingSeekRef = useRef<number | null>(null);
  /**
   * When streaming completes, `onComplete` in CreateView dispatches
   * SET_STREAMING false and PLAY_TRACK (library track).  But we may still
   * be inside an ended handler.  This flag lets us know we should NOT
   * try to mark streaming as over from the ended handler.
   */
  const streamingCompleteRef = useRef(false);

  // ---- Single ref mirror for values needed inside stable (registered-once)
  //      event handlers.  Updated every render — always fresh. ----
  const stateRef = useRef({
    isStreaming: state.isStreaming,
    isPlaying: state.isPlaying,
    repeatMode,
    volume,
    isMuted,
  });
  stateRef.current = {
    isStreaming: state.isStreaming,
    isPlaying: state.isPlaying,
    repeatMode,
    volume,
    isMuted,
  };

  // ---- Volume helper ----
  const applyVolume = useCallback((el: HTMLAudioElement) => {
    const s = stateRef.current;
    el.volume = s.isMuted ? 0 : s.volume / 100;
  }, []);

  // ---- Preload next segment onto standby element ----
  const preloadNext = useCallback(() => {
    const segs = streamBus.segments;
    const nextIdx = streamIndexRef.current + 1;
    if (nextIdx >= segs.length) return;
    if (preloadedIndexRef.current === nextIdx) return;

    const standby = getStandby();
    if (!standby) return;

    preloadedIndexRef.current = nextIdx;
    standby.src = segs[nextIdx];
    standby.preload = "auto";
    applyVolume(standby);
  }, [getStandby, applyVolume]);

  // ---- Swap active <-> standby and start playing the next segment ----
  const swapToStandby = useCallback(() => {
    const segs = streamBus.segments;
    const nextIdx = streamIndexRef.current + 1;
    if (nextIdx >= segs.length) return;

    const active = getActive();
    const standby = getStandby();
    if (!active || !standby) return;

    // Accumulate time from the segment that just finished
    streamBaseTimeRef.current += active.duration || 0;

    // Flip the active pointer
    activeRef.current = activeRef.current === "A" ? "B" : "A";
    streamIndexRef.current = nextIdx;
    preloadedIndexRef.current = -1;
    waitingForSegmentRef.current = false;

    applyVolume(standby);
    standby.play().catch(() => {});

    // Try to preload the one after
    preloadNext();
  }, [getActive, getStandby, applyVolume, preloadNext]);

  // Stable ref so that the ended handler (registered once) always calls
  // the latest closure.
  const swapRef = useRef(swapToStandby);
  swapRef.current = swapToStandby;

  // ---- Library navigation (needs to be defined before the ended handler) ----
  const handleNext = useCallback(() => {
    if (stateRef.current.isStreaming) return;
    const idx = state.songs.findIndex(
      (s) => s.filename === state.currentTrack?.filename,
    );
    if (idx < state.songs.length - 1) {
      dispatch({ type: "PLAY_TRACK", track: state.songs[idx + 1] });
    } else if (repeatMode === "all" && state.songs.length > 0) {
      dispatch({ type: "PLAY_TRACK", track: state.songs[0] });
    }
  }, [state.songs, state.currentTrack, repeatMode, dispatch]);

  const handleNextRef = useRef(handleNext);
  handleNextRef.current = handleNext;

  // ------------------------------------------------------------------
  // Effect: Sync volume to both elements whenever volume/muted changes
  // ------------------------------------------------------------------
  useEffect(() => {
    const v = isMuted ? 0 : volume / 100;
    if (audioARef.current) audioARef.current.volume = v;
    if (audioBRef.current) audioBRef.current.volume = v;
  }, [volume, isMuted]);

  // ------------------------------------------------------------------
  // Effect: Load a normal (non-streaming) library track on audioA
  // ------------------------------------------------------------------
  useEffect(() => {
    // GUARD: do not touch audio elements while we are streaming.
    if (state.isStreaming) return;

    const audio = audioARef.current;
    if (!audio || !state.currentTrack) return;

    // Kill BOTH elements to ensure no leftover streaming playback.
    killAudio(audioBRef.current);
    activeRef.current = "A";
    streamIndexRef.current = -1;
    streamBaseTimeRef.current = 0;
    preloadedIndexRef.current = -1;
    waitingForSegmentRef.current = false;

    pendingSeekRef.current = state.seekTo;

    audio.src = api.audioUrl(state.currentTrack.filename);
    applyVolume(audio);
    audio.play().catch(() => {});

    if (state.seekTo !== null) {
      dispatch({ type: "CLEAR_SEEK" });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.currentTrack, state.isStreaming]);

  // ------------------------------------------------------------------
  // Effect: Subscribe to streamBus for segment notifications (once)
  // ------------------------------------------------------------------
  useEffect(() => {
    const audioA = audioARef.current;
    if (!audioA) return;

    streamBus.onSegment = () => {
      const segs = streamBus.segments;

      // --- First segment: kill any library playback, start streaming ---
      if (streamIndexRef.current === -1 && segs.length > 0) {
        // Exclusive ownership: kill both elements first
        killAudio(audioARef.current);
        killAudio(audioBRef.current);

        activeRef.current = "A";
        streamIndexRef.current = 0;
        streamBaseTimeRef.current = 0;
        preloadedIndexRef.current = -1;
        waitingForSegmentRef.current = false;
        streamingCompleteRef.current = false;

        const el = audioARef.current!;
        el.src = segs[0];
        applyVolume(el);
        el.play().catch(() => {});
        return;
      }

      // --- We were waiting for the next segment and it just arrived ---
      if (waitingForSegmentRef.current) {
        const nextIdx = streamIndexRef.current + 1;
        if (nextIdx < segs.length) {
          swapRef.current();
          return;
        }
      }

      // --- Active element already ended while we had no next segment ---
      const active =
        activeRef.current === "A" ? audioARef.current : audioBRef.current;
      if (active?.ended) {
        const nextIdx = streamIndexRef.current + 1;
        if (nextIdx < segs.length) {
          swapRef.current();
          return;
        }
      }

      // --- Normal case: just try to preload the next segment ---
      preloadNext();
    };

    return () => {
      streamBus.onSegment = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ------------------------------------------------------------------
  // Effect: When streaming stops, reset streaming indices
  // ------------------------------------------------------------------
  useEffect(() => {
    if (!state.isStreaming) {
      streamIndexRef.current = -1;
      streamBaseTimeRef.current = 0;
      preloadedIndexRef.current = -1;
      waitingForSegmentRef.current = false;
      activeRef.current = "A";
    }
  }, [state.isStreaming]);

  // ------------------------------------------------------------------
  // Effect: "ended" handler — registered ONCE on both elements
  // ------------------------------------------------------------------
  useEffect(() => {
    const a = audioARef.current;
    const b = audioBRef.current;
    if (!a || !b) return;

    const onEnded = (e: Event) => {
      const el = e.currentTarget as HTMLAudioElement;
      // Ignore events from the standby element
      const activeEl =
        activeRef.current === "A" ? audioARef.current : audioBRef.current;
      if (el !== activeEl) return;

      if (stateRef.current.isStreaming) {
        const segs = streamBus.segments;
        const nextIdx = streamIndexRef.current + 1;

        if (nextIdx < segs.length) {
          // Next segment is already available — swap immediately.
          swapRef.current();
        } else {
          // Next segment hasn't arrived yet.
          // DO NOT dispatch SET_STREAMING false — instead, pause and wait.
          // The onSegment callback will resume us when the segment arrives.
          waitingForSegmentRef.current = true;
          // If streaming has actually completed (no more segments coming),
          // CreateView's onComplete will handle the transition.
        }
        return;
      }

      // ---- Normal library playback ----
      if (stateRef.current.repeatMode === "one") {
        el.currentTime = 0;
        el.play();
      } else {
        handleNextRef.current();
      }
    };

    a.addEventListener("ended", onEnded);
    b.addEventListener("ended", onEnded);
    return () => {
      a.removeEventListener("ended", onEnded);
      b.removeEventListener("ended", onEnded);
    };
  }, []); // registered once

  // ------------------------------------------------------------------
  // Effect: timeupdate + loadedmetadata — registered ONCE on both
  // ------------------------------------------------------------------
  useEffect(() => {
    const a = audioARef.current;
    const b = audioBRef.current;
    if (!a || !b) return;

    const onTimeUpdate = (e: Event) => {
      const el = e.currentTarget as HTMLAudioElement;
      const activeEl =
        activeRef.current === "A" ? audioARef.current : audioBRef.current;
      if (el !== activeEl) return;

      if (stateRef.current.isStreaming) {
        const t = streamBaseTimeRef.current + el.currentTime;
        setCurrentTime(t);
        // Write to bus imperatively — no dispatch, no re-render.
        streamBus.time = t;
      } else {
        setCurrentTime(el.currentTime);
      }
    };

    const onLoadedMetadata = (e: Event) => {
      const el = e.currentTarget as HTMLAudioElement;
      const activeEl =
        activeRef.current === "A" ? audioARef.current : audioBRef.current;
      if (el !== activeEl) return;

      if (!stateRef.current.isStreaming) {
        setTotalDuration(el.duration);
        const seekPos = pendingSeekRef.current;
        if (seekPos !== null && seekPos > 0 && seekPos < el.duration) {
          el.currentTime = seekPos;
          pendingSeekRef.current = null;
        }
      }
    };

    for (const el of [a, b]) {
      el.addEventListener("timeupdate", onTimeUpdate);
      el.addEventListener("loadedmetadata", onLoadedMetadata);
    }
    return () => {
      for (const el of [a, b]) {
        el.removeEventListener("timeupdate", onTimeUpdate);
        el.removeEventListener("loadedmetadata", onLoadedMetadata);
      }
    };
  }, []); // registered once

  // ------------------------------------------------------------------
  // Effect: Sync play/pause from state to the active element
  // ------------------------------------------------------------------
  useEffect(() => {
    const active = getActive();
    if (!active) return;
    // Don't interfere if we are waiting for a segment — there's nothing
    // to play right now; the onSegment callback will resume.
    if (waitingForSegmentRef.current) return;

    if (state.isPlaying) {
      active.play().catch(() => {});
    } else {
      active.pause();
    }
  }, [state.isPlaying, getActive]);

  // ------------------------------------------------------------------
  // Handlers
  // ------------------------------------------------------------------

  const handlePrev = useCallback(() => {
    if (stateRef.current.isStreaming) return;
    const audio = getActive();
    if (audio && audio.currentTime > 3) {
      audio.currentTime = 0;
      return;
    }
    const idx = state.songs.findIndex(
      (s) => s.filename === state.currentTrack?.filename,
    );
    if (idx > 0) {
      dispatch({ type: "PLAY_TRACK", track: state.songs[idx - 1] });
    }
  }, [state.songs, state.currentTrack, dispatch, getActive]);

  const togglePlay = () => {
    dispatch({ type: "SET_PLAYING", isPlaying: !state.isPlaying });
  };

  const handleSeek = (value: number[]) => {
    const audio = getActive();
    if (audio && totalDuration && !state.isStreaming) {
      audio.currentTime = (value[0] / 100) * totalDuration;
    }
  };

  const handleVolumeChange = (value: number[]) => {
    const v = value[0];
    setVolume(v);
    setIsMuted(v === 0);
  };

  const toggleMute = () => {
    setIsMuted((prev) => !prev);
  };

  const handleDownload = () => {
    if (!state.currentTrack) return;
    const ext = state.currentTrack.filename.endsWith(".mp3") ? ".mp3" : ".wav";
    const a = document.createElement("a");
    a.href = api.audioUrl(state.currentTrack.filename);
    a.download = (state.currentTrack.title || "track") + ext;
    a.click();
  };

  // ------------------------------------------------------------------
  // Render
  // ------------------------------------------------------------------

  if (!state.currentTrack && !state.isStreaming) {
    return (
      <>
        <audio ref={audioARef} />
        <audio ref={audioBRef} />
      </>
    );
  }

  const progressPercent = totalDuration
    ? (currentTime / totalDuration) * 100
    : 0;

  return (
    <>
      <audio ref={audioARef} />
      <audio ref={audioBRef} />
      <div className="fixed bottom-0 left-0 right-0 z-50 flex h-22.5 items-center gap-4 border-t border-white/5 bg-linear-to-b from-[#181818] to-[#121212] px-4">
        <div className="flex w-55 min-w-55 items-center gap-3.5">
          <div
            className="h-14 w-14 shrink-0 rounded shadow-lg"
            style={getAlbumArtStyle(state.currentTrack)}
          />
          <div className="min-w-0">
            <div className="truncate text-sm font-medium">
              {state.currentTrack?.title ??
                (state.isStreaming ? "Generating..." : "")}
            </div>
            <div className="truncate text-xs text-muted-foreground">
              {state.isStreaming ? "Streaming preview" : "HeartMuLa"}
            </div>
          </div>
        </div>

        <div className="mx-auto flex max-w-180 flex-1 flex-col items-center gap-2">
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="icon"
              className={cn("h-8 w-8", isShuffled && "text-green-500")}
              onClick={() => setIsShuffled(!isShuffled)}
            >
              <Shuffle className="h-4 w-4" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8"
              onClick={handlePrev}
            >
              <SkipBack className="h-5 w-5" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="h-10 w-10 rounded-full bg-white text-black hover:scale-105 hover:bg-white"
              onClick={togglePlay}
            >
              {state.isPlaying ? (
                <Pause className="h-5 w-5" />
              ) : (
                <Play className="ml-0.5 h-5 w-5" />
              )}
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8"
              onClick={handleNext}
            >
              <SkipForward className="h-5 w-5" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className={cn(
                "h-8 w-8",
                repeatMode !== "none" && "text-green-500",
              )}
              onClick={() => {
                const modes: Array<"none" | "all" | "one"> = [
                  "none",
                  "all",
                  "one",
                ];
                const idx = modes.indexOf(repeatMode);
                setRepeatMode(modes[(idx + 1) % modes.length]);
              }}
            >
              <Repeat className="h-4 w-4" />
              {repeatMode === "one" && (
                <span className="absolute text-[8px] font-bold">1</span>
              )}
            </Button>
          </div>

          <div className="flex w-full items-center gap-2.5">
            <span className="min-w-10 text-right text-[11px] tabular-nums text-muted-foreground">
              {formatDuration(currentTime)}
            </span>
            <Slider
              value={[progressPercent]}
              onValueChange={handleSeek}
              max={100}
              step={0.1}
              className="flex-1"
            />
            <span className="min-w-10 text-[11px] tabular-nums text-muted-foreground">
              {state.isStreaming ? "..." : formatDuration(totalDuration)}
            </span>
          </div>
        </div>

        {/* Right: Actions + volume */}
        <div className="flex w-55 min-w-55 items-center justify-end gap-1">
          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8"
            onClick={handleDownload}
          >
            <Download className="h-4 w-4" />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8"
            onClick={toggleMute}
          >
            {isMuted ? (
              <VolumeX className="h-4 w-4" />
            ) : (
              <Volume2 className="h-4 w-4" />
            )}
          </Button>
          <Slider
            value={[isMuted ? 0 : volume]}
            onValueChange={handleVolumeChange}
            max={100}
            step={1}
            className="w-20"
          />
        </div>
      </div>
    </>
  );
}

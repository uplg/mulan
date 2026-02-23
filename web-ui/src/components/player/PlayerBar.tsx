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

export function PlayerBar() {
  const { state, dispatch } = useApp();

  // ---- Two <audio> elements for gapless double-buffering. ----
  const audioARef = useRef<HTMLAudioElement>(null);
  const audioBRef = useRef<HTMLAudioElement>(null);
  /** Which element is currently active: "A" | "B". */
  const activeRef = useRef<"A" | "B">("A");

  const getActive = () =>
    activeRef.current === "A" ? audioARef.current : audioBRef.current;
  const getStandby = () =>
    activeRef.current === "A" ? audioBRef.current : audioARef.current;

  const [currentTime, setCurrentTime] = useState(0);
  const [totalDuration, setTotalDuration] = useState(0);
  const [volume, setVolume] = useState(80);
  const [isMuted, setIsMuted] = useState(false);
  const [isShuffled, setIsShuffled] = useState(false);
  const [repeatMode, setRepeatMode] = useState<"none" | "all" | "one">("none");

  // ---- Refs for stable access from event handlers registered once. ----
  const streamIndexRef = useRef(-1);
  const preloadedIndexRef = useRef(-1);
  const streamBaseTimeRef = useRef(0);
  const repeatModeRef = useRef(repeatMode);
  repeatModeRef.current = repeatMode;
  const isStreamingRef = useRef(state.isStreaming);
  isStreamingRef.current = state.isStreaming;
  const volumeRef = useRef(volume);
  volumeRef.current = volume;
  const isMutedRef = useRef(isMuted);
  isMutedRef.current = isMuted;
  /** Pending seek position — consumed once after the next loadedmetadata event. */
  const pendingSeekRef = useRef<number | null>(null);

  // ---- Volume helper ----
  const applyVolume = (el: HTMLAudioElement) => {
    el.volume = isMutedRef.current ? 0 : volumeRef.current / 100;
  };

  // ---- Preload next segment onto standby element. ----
  const preloadNext = () => {
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
  };

  // ---- Swap active ↔ standby. ----
  const swapToStandby = () => {
    const segs = streamBus.segments;
    const nextIdx = streamIndexRef.current + 1;
    if (nextIdx >= segs.length) return;

    const active = getActive();
    const standby = getStandby();
    if (!active || !standby) return;

    streamBaseTimeRef.current += active.duration || 0;
    activeRef.current = activeRef.current === "A" ? "B" : "A";
    streamIndexRef.current = nextIdx;
    preloadedIndexRef.current = -1;

    applyVolume(standby);
    standby.play().catch(() => {});

    preloadNext();
  };

  // Store swap in a ref so the ended listener (registered once) always
  // calls the latest version without needing to re-subscribe.
  const swapRef = useRef(swapToStandby);
  swapRef.current = swapToStandby;

  // ---- Sync volume to both elements. ----
  useEffect(() => {
    const v = isMuted ? 0 : volume / 100;
    if (audioARef.current) audioARef.current.volume = v;
    if (audioBRef.current) audioBRef.current.volume = v;
  }, [volume, isMuted]);

  // ---- Load a normal (non-streaming) track on audioA. ----
  useEffect(() => {
    const audio = audioARef.current;
    if (!audio || !state.currentTrack) return;
    if (state.isStreaming) return;
    activeRef.current = "A";
    streamIndexRef.current = -1;
    streamBaseTimeRef.current = 0;
    preloadedIndexRef.current = -1;
    pendingSeekRef.current = state.seekTo;
    audio.src = api.audioUrl(state.currentTrack.filename);
    applyVolume(audio);
    audio.play().catch(() => {});
    // Clear seekTo from state so it doesn't re-trigger
    if (state.seekTo !== null) {
      dispatch({ type: "CLEAR_SEEK" });
    }
  }, [state.currentTrack, state.isStreaming]);

  // ---- Subscribe to streamBus for segment notifications. ----
  // Registered once. The callback reads streamBus.segments imperatively.
  useEffect(() => {
    const audio = audioARef.current;
    if (!audio) return;

    streamBus.onSegment = () => {
      const segs = streamBus.segments;

      // First segment — start playback on audioA.
      if (streamIndexRef.current === -1 && segs.length > 0) {
        activeRef.current = "A";
        streamIndexRef.current = 0;
        streamBaseTimeRef.current = 0;
        preloadedIndexRef.current = -1;
        audio.src = segs[0];
        applyVolume(audio);
        audio.play().catch(() => {});
      }

      // If the active element has ended (segment finished before next
      // arrived from the server), swap now.
      const active = getActive();
      if (active?.ended) {
        const nextIdx = streamIndexRef.current + 1;
        if (nextIdx < segs.length) {
          swapRef.current();
          return;
        }
      }

      // Otherwise just try to preload the next segment onto standby.
      preloadNext();
    };

    return () => {
      streamBus.onSegment = null;
    };
  }, []);

  // ---- When streaming stops (state.isStreaming false), reset indices. ----
  useEffect(() => {
    if (!state.isStreaming) {
      streamIndexRef.current = -1;
      streamBaseTimeRef.current = 0;
      preloadedIndexRef.current = -1;
      activeRef.current = "A";
    }
  }, [state.isStreaming]);

  // ---- Single "ended" handler — registered ONCE, uses refs. ----
  useEffect(() => {
    const a = audioARef.current;
    const b = audioBRef.current;
    if (!a || !b) return;

    const onEnded = (e: Event) => {
      const el = e.currentTarget as HTMLAudioElement;
      if (el !== getActive()) return; // ignore stale standby events

      if (isStreamingRef.current) {
        const segs = streamBus.segments;
        const nextIdx = streamIndexRef.current + 1;
        if (nextIdx < segs.length) {
          swapRef.current();
        } else {
          // Last segment just finished — streaming playback is over.
          // CreateView's onComplete already handles the transition to
          // the final track; if it hasn't fired yet, it will find
          // isStreaming = false and handle accordingly.
          dispatch({ type: "SET_STREAMING", isStreaming: false });
        }
        return;
      }

      // Normal library playback.
      if (repeatModeRef.current === "one") {
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
  }, []); // truly registered once — never re-subscribes

  // ---- Time / metadata — listen on BOTH, report from active only. ----
  useEffect(() => {
    const a = audioARef.current;
    const b = audioBRef.current;
    if (!a || !b) return;

    const onTimeUpdate = (e: Event) => {
      const el = e.currentTarget as HTMLAudioElement;
      if (el !== getActive()) return;
      if (isStreamingRef.current) {
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
      if (el !== getActive()) return;
      if (!isStreamingRef.current) {
        setTotalDuration(el.duration);
        // If a seek was requested (e.g. seamless transition from streaming),
        // apply it now that metadata is available.
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
  }, []);

  // ---- Sync play/pause ----
  useEffect(() => {
    const active = getActive();
    if (!active) return;
    if (state.isPlaying) {
      active.play().catch(() => {});
    } else {
      active.pause();
    }
  }, [state.isPlaying]);

  // ---- Library navigation ----
  const handleNext = useCallback(() => {
    if (isStreamingRef.current) return;
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

  const handlePrev = useCallback(() => {
    if (isStreamingRef.current) return;
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
  }, [state.songs, state.currentTrack, dispatch]);

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

  if (!state.currentTrack && !state.isStreaming) {
    return (
      <>
        <audio ref={audioARef} />
        <audio ref={audioBRef} />
      </>
    );
  }

  const progressPercent = totalDuration ? (currentTime / totalDuration) * 100 : 0;

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
              {state.currentTrack?.title ?? (state.isStreaming ? "Generating..." : "")}
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
            <Button variant="ghost" size="icon" className="h-8 w-8" onClick={handlePrev}>
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
                <Play className="h-5 w-5 ml-0.5" />
              )}
            </Button>
            <Button variant="ghost" size="icon" className="h-8 w-8" onClick={handleNext}>
              <SkipForward className="h-5 w-5" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className={cn("h-8 w-8", repeatMode !== "none" && "text-green-500")}
              onClick={() => {
                const modes: Array<"none" | "all" | "one"> = ["none", "all", "one"];
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
          <Button variant="ghost" size="icon" className="h-8 w-8" onClick={handleDownload}>
            <Download className="h-4 w-4" />
          </Button>
          <Button variant="ghost" size="icon" className="h-8 w-8" onClick={toggleMute}>
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

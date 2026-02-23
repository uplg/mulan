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
import { getAlbumArtStyle, formatDuration } from "@/lib/helpers";
import { cn } from "@/lib/utils";

export function PlayerBar() {
  const { state, dispatch } = useApp();
  const audioRef = useRef<HTMLAudioElement>(null);

  const [currentTime, setCurrentTime] = useState(0);
  const [totalDuration, setTotalDuration] = useState(0);
  const [volume, setVolume] = useState(80);
  const [isMuted, setIsMuted] = useState(false);
  const [isShuffled, setIsShuffled] = useState(false);
  const [repeatMode, setRepeatMode] = useState<"none" | "all" | "one">("none");

  const isStreaming = state.streamingSegments.length > 0;

  // ---- Refs that let the single "ended" handler read fresh values
  //      without re-subscribing the listener on every render. ----
  const streamSegmentsRef = useRef(state.streamingSegments);
  streamSegmentsRef.current = state.streamingSegments;

  /** Index of the segment currently loaded in <audio> (-1 = none). */
  const streamIndexRef = useRef(-1);

  /** Cumulative playback time of all *completed* segments. */
  const streamBaseTimeRef = useRef(0);

  const repeatModeRef = useRef(repeatMode);
  repeatModeRef.current = repeatMode;

  const isStreamingRef = useRef(isStreaming);
  isStreamingRef.current = isStreaming;

  // ---- Load a normal (non-streaming) track ----
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio || !state.currentTrack) return;
    if (isStreaming) return;
    streamIndexRef.current = -1;
    streamBaseTimeRef.current = 0;
    audio.src = api.audioUrl(state.currentTrack.filename);
    audio.play().catch(() => {});
  }, [state.currentTrack, isStreaming]);

  // ---- Streaming: load first segment once ----
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    if (state.streamingSegments.length === 0) {
      streamIndexRef.current = -1;
      streamBaseTimeRef.current = 0;
      return;
    }

    // Load the very first segment only once.
    if (streamIndexRef.current === -1) {
      streamIndexRef.current = 0;
      streamBaseTimeRef.current = 0;
      audio.src = state.streamingSegments[0];
      audio.play().catch(() => {});
    }
  }, [state.streamingSegments]);

  // ---- Streaming: when a segment finishes but the next wasn't available
  //      yet, this effect detects the new segment arrived and loads it. ----
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio || !isStreaming) return;

    const idx = streamIndexRef.current;
    if (idx < 0) return;

    const nextIdx = idx + 1;
    const haveNext = nextIdx < state.streamingSegments.length;
    // audio.ended is true when the segment played through completely.
    if (haveNext && audio.ended) {
      streamBaseTimeRef.current += audio.duration || 0;
      streamIndexRef.current = nextIdx;
      audio.src = state.streamingSegments[nextIdx];
      audio.play().catch(() => {});
    }
  }, [state.streamingSegments, isStreaming]);

  // ---- Single "ended" handler — registered ONCE, reads refs. ----
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    const onEnded = () => {
      if (isStreamingRef.current) {
        // Streaming mode: advance to next segment if available.
        const segs = streamSegmentsRef.current;
        const nextIdx = streamIndexRef.current + 1;
        if (nextIdx < segs.length) {
          streamBaseTimeRef.current += audio.duration || 0;
          streamIndexRef.current = nextIdx;
          audio.src = segs[nextIdx];
          audio.play().catch(() => {});
        }
        // else: wait — the "new segment arrived" effect above will pick it up.
        return;
      }

      // Normal library playback
      if (repeatModeRef.current === "one") {
        audio.currentTime = 0;
        audio.play();
      } else {
        // handleNext is expensive to close over stably, so we dispatch
        // a lightweight action and let the next-track effect below handle it.
        handleNextRef.current();
      }
    };

    audio.addEventListener("ended", onEnded);
    return () => audio.removeEventListener("ended", onEnded);
  }, []); // registered once, never re-subscribed

  // ---- Time / metadata updates ----
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    const onTimeUpdate = () => {
      if (isStreamingRef.current) {
        setCurrentTime(streamBaseTimeRef.current + audio.currentTime);
      } else {
        setCurrentTime(audio.currentTime);
      }
    };

    const onLoadedMetadata = () => {
      if (!isStreamingRef.current) {
        setTotalDuration(audio.duration);
      }
      // During streaming we don't update totalDuration per-segment
      // because we don't know the full duration yet.
    };

    audio.addEventListener("timeupdate", onTimeUpdate);
    audio.addEventListener("loadedmetadata", onLoadedMetadata);
    return () => {
      audio.removeEventListener("timeupdate", onTimeUpdate);
      audio.removeEventListener("loadedmetadata", onLoadedMetadata);
    };
  }, []);

  // ---- Sync play/pause ----
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;
    if (state.isPlaying) {
      audio.play().catch(() => {});
    } else {
      audio.pause();
    }
  }, [state.isPlaying]);

  // ---- Library navigation ----
  const handleNext = useCallback(() => {
    if (isStreamingRef.current) return; // don't skip during streaming
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
    const audio = audioRef.current;
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
    const audio = audioRef.current;
    if (audio && totalDuration && !isStreaming) {
      audio.currentTime = (value[0] / 100) * totalDuration;
    }
  };

  const handleVolumeChange = (value: number[]) => {
    const v = value[0];
    setVolume(v);
    setIsMuted(v === 0);
    if (audioRef.current) audioRef.current.volume = v / 100;
  };

  const toggleMute = () => {
    const audio = audioRef.current;
    if (!audio) return;
    if (isMuted) {
      audio.volume = volume / 100;
      setIsMuted(false);
    } else {
      audio.volume = 0;
      setIsMuted(true);
    }
  };

  const handleDownload = () => {
    if (!state.currentTrack) return;
    const ext = state.currentTrack.filename.endsWith(".mp3") ? ".mp3" : ".wav";
    const a = document.createElement("a");
    a.href = api.audioUrl(state.currentTrack.filename);
    a.download = (state.currentTrack.title || "track") + ext;
    a.click();
  };

  if (!state.currentTrack && !isStreaming) return <audio ref={audioRef} />;

  const progressPercent = totalDuration ? (currentTime / totalDuration) * 100 : 0;

  return (
    <>
      <audio ref={audioRef} />
      <div className="fixed bottom-0 left-0 right-0 z-50 flex h-22.5 items-center gap-4 border-t border-white/5 bg-linear-to-b from-[#181818] to-[#121212] px-4">
        <div className="flex w-55 min-w-55 items-center gap-3.5">
          <div
            className="h-14 w-14 shrink-0 rounded shadow-lg"
            style={getAlbumArtStyle(state.currentTrack)}
          />
          <div className="min-w-0">
            <div className="truncate text-sm font-medium">
              {state.currentTrack?.title ?? (isStreaming ? "Generating..." : "")}
            </div>
            <div className="truncate text-xs text-muted-foreground">
              {isStreaming ? "Streaming preview" : "HeartMuLa"}
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
              {isStreaming ? "..." : formatDuration(totalDuration)}
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

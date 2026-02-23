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

  // Load track when currentTrack changes
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio || !state.currentTrack) return;
    audio.src = api.audioUrl(state.currentTrack.filename);
    audio.play().catch(() => {});
  }, [state.currentTrack]);

  // Sync play/pause state
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;
    if (state.isPlaying) {
      audio.play().catch(() => {});
    } else {
      audio.pause();
    }
  }, [state.isPlaying]);

  const handleNext = useCallback(() => {
    const idx = state.songs.findIndex(
      (s) => s.filename === state.currentTrack?.filename
    );
    if (idx < state.songs.length - 1) {
      dispatch({ type: "PLAY_TRACK", track: state.songs[idx + 1] });
    } else if (repeatMode === "all" && state.songs.length > 0) {
      dispatch({ type: "PLAY_TRACK", track: state.songs[0] });
    }
  }, [state.songs, state.currentTrack, repeatMode, dispatch]);

  // Audio events
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    const onTimeUpdate = () => setCurrentTime(audio.currentTime);
    const onLoadedMetadata = () => setTotalDuration(audio.duration);
    const onEnded = () => {
      if (repeatMode === "one") {
        audio.currentTime = 0;
        audio.play();
      } else {
        handleNext();
      }
    };

    audio.addEventListener("timeupdate", onTimeUpdate);
    audio.addEventListener("loadedmetadata", onLoadedMetadata);
    audio.addEventListener("ended", onEnded);

    return () => {
      audio.removeEventListener("timeupdate", onTimeUpdate);
      audio.removeEventListener("loadedmetadata", onLoadedMetadata);
      audio.removeEventListener("ended", onEnded);
    };
  }, [repeatMode, handleNext]);

  const togglePlay = () => {
    dispatch({ type: "SET_PLAYING", isPlaying: !state.isPlaying });
  };

  const handleSeek = (value: number[]) => {
    const audio = audioRef.current;
    if (audio && totalDuration) {
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

  const handlePrev = useCallback(() => {
    const audio = audioRef.current;
    if (audio && audio.currentTime > 3) {
      audio.currentTime = 0;
      return;
    }
    const idx = state.songs.findIndex(
      (s) => s.filename === state.currentTrack?.filename
    );
    if (idx > 0) {
      dispatch({ type: "PLAY_TRACK", track: state.songs[idx - 1] });
    }
  }, [state.songs, state.currentTrack, dispatch]);


  const handleDownload = () => {
    if (!state.currentTrack) return;
    const ext = state.currentTrack.filename.endsWith(".mp3") ? ".mp3" : ".wav";
    const a = document.createElement("a");
    a.href = api.audioUrl(state.currentTrack.filename);
    a.download = (state.currentTrack.title || "track") + ext;
    a.click();
  };

  if (!state.currentTrack) return <audio ref={audioRef} />;

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
              {state.currentTrack.title}
            </div>
            <div className="truncate text-xs text-muted-foreground">HeartMuLa</div>
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
              {formatDuration(totalDuration)}
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

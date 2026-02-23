import { useState, useCallback, useRef, useEffect } from "react";
import { Sparkles, ChevronDown, Wand2, Loader2, Music } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { useApp } from "@/lib/store";
import { api } from "@/lib/api";
import { cn } from "@/lib/utils";

const STYLE_TAGS = [
  "pop", "rock", "jazz", "piano", "acoustic", "electronic", "female vocal", "male vocal",
];

export function CreateView() {
  const { state, dispatch } = useApp();

  const [lyrics, setLyrics] = useState("");
  const [styles, setStyles] = useState("electronic,ambient,instrumental");
  const [title, setTitle] = useState("");
  const [duration, setDuration] = useState(30);
  const [temperature, setTemperature] = useState(1.0);
  const [cfgScale, setCfgScale] = useState(1.5);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [lyricsOpen, setLyricsOpen] = useState(true);
  const [stylesOpen, setStylesOpen] = useState(true);

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const generateAbortRef = useRef<AbortController | null>(null);

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  useEffect(() => {
    return () => {
      stopPolling();
      generateAbortRef.current?.abort();
    };
  }, [stopPolling]);

  const addTag = (tag: string) => {
    const current = styles.split(",").map((t) => t.trim()).filter(Boolean);
    if (!current.includes(tag)) {
      setStyles([...current, tag].join(","));
    }
  };

  const surpriseMe = useCallback(async () => {
    try {
      const data = await api.getSurpriseLyrics();
      setLyrics(data.lyrics);
      setStyles(data.styles);
      setTitle(data.title);
    } catch (e) {
      console.error("Surprise failed:", e);
    }
  }, []);

  const startProgressPolling = useCallback(() => {
    if (pollRef.current) return;
    pollRef.current = setInterval(async () => {
      try {
        const status = await api.getProgress();
        dispatch({
          type: "SET_PROGRESS",
          progress: status.progress,
          message: status.message,
        });
        if (!status.is_generating) {
          stopPolling();
          dispatch({ type: "SET_GENERATING", isGenerating: false });
          // Reload songs
          const data = await api.getSongs();
          dispatch({ type: "SET_SONGS", songs: data.songs });
        }
      } catch {
        // Ignore transient polling errors
      }
    }, 1000);
  }, [dispatch, stopPolling]);

  const handleGenerate = useCallback(async () => {
    if (state.isGenerating) return;

    dispatch({ type: "SET_GENERATING", isGenerating: true });
    dispatch({ type: "SET_PROGRESS", progress: 0, message: "Starting..." });

    // Create a dedicated AbortController for the generate fetch
    const controller = new AbortController();
    generateAbortRef.current = controller;

    startProgressPolling();

    try {
      const result = await api.generate(
        {
          lyrics,
          styles,
          duration,
          cfg_scale: cfgScale,
          temperature,
          topk: 50,
          title: title || `Generation ${state.songs.length + 1}`,
          language: "",
        },
        controller.signal,
      );

      // Generate completed successfully — stop polling, update state
      stopPolling();
      dispatch({ type: "SET_GENERATING", isGenerating: false });
      dispatch({ type: "SET_PROGRESS", progress: 1, message: "Complete!" });

      // Reload songs and auto-play the new track
      const data = await api.getSongs();
      dispatch({ type: "SET_SONGS", songs: data.songs });
      const newSong = data.songs.find((s) => s.filename === result.filename);
      if (newSong) {
        dispatch({ type: "PLAY_TRACK", track: newSong });
      }
    } catch (error) {
      // If aborted (cancel), let polling handle the final state reset.
      // The server-side cancel sets is_generating=false after the pipeline
      // stops, so polling will pick that up.
      if (error instanceof DOMException && error.name === "AbortError") {
        // Aborted by cancel — polling will clean up
        return;
      }
      // Actual error — stop polling and show error
      stopPolling();
      dispatch({ type: "SET_GENERATING", isGenerating: false });
      dispatch({
        type: "SET_PROGRESS",
        progress: 0,
        message: `Error: ${error instanceof Error ? error.message : "Unknown error"}`,
      });
    }
  }, [state.isGenerating, state.songs.length, lyrics, styles, duration, cfgScale, temperature, title, dispatch, startProgressPolling, stopPolling]);

  const handleCancel = useCallback(async () => {
    try {
      // Tell server to cancel the pipeline
      await api.cancel();
      dispatch({ type: "SET_PROGRESS", progress: state.progress, message: "Cancelling..." });
      // Abort the long-running generate fetch so handleGenerate doesn't
      // try to process a (now-invalid) response
      generateAbortRef.current?.abort();
    } catch (e) {
      console.error("Failed to cancel:", e);
    }
  }, [dispatch, state.progress]);

  return (
    <div className="flex flex-col gap-2">
      <section>
        <button
          className="flex w-full items-center gap-2 py-3"
          onClick={() => setLyricsOpen(!lyricsOpen)}
        >
          <ChevronDown
            className={cn(
              "h-5 w-5 text-muted-foreground transition-transform",
              !lyricsOpen && "-rotate-90"
            )}
          />
          <span className="text-[15px] font-medium">Lyrics</span>
          <Button
            variant="ghost"
            size="icon"
            className="ml-auto h-7 w-7 text-muted-foreground hover:text-foreground"
            onClick={(e) => {
              e.stopPropagation();
              surpriseMe();
            }}
            title="Surprise Me!"
          >
            <Wand2 className="h-4 w-4" />
          </Button>
        </button>
        {lyricsOpen && (
          <Textarea
            value={lyrics}
            onChange={(e) => setLyrics(e.target.value)}
            placeholder="Write some lyrics or a prompt -- or leave blank for instrumental"
            rows={6}
            className="resize-none rounded-xl border-border bg-input text-sm leading-relaxed"
          />
        )}
      </section>

      <section>
        <button
          className="flex w-full items-center gap-2 py-3"
          onClick={() => setStylesOpen(!stylesOpen)}
        >
          <ChevronDown
            className={cn(
              "h-5 w-5 text-muted-foreground transition-transform",
              !stylesOpen && "-rotate-90"
            )}
          />
          <span className="text-[15px] font-medium">Styles</span>
        </button>
        {stylesOpen && (
          <div>
            <Input
              value={styles}
              onChange={(e) => setStyles(e.target.value)}
              placeholder="pop,rock,electronic,piano,female vocal"
              className="rounded-xl border-border bg-input text-sm"
            />
            <div className="mt-4 flex flex-wrap gap-2">
              {STYLE_TAGS.map((tag) => (
                <Badge
                  key={tag}
                  variant="outline"
                  className="cursor-pointer rounded-full px-3.5 py-2 text-[13px] font-normal transition-colors hover:bg-accent"
                  onClick={() => addTag(tag)}
                >
                  + {tag}
                </Badge>
              ))}
            </div>
          </div>
        )}
      </section>

      <section>
        <button
          className="flex w-full items-center gap-2 py-3"
          onClick={() => setAdvancedOpen(!advancedOpen)}
        >
          <ChevronDown
            className={cn(
              "h-5 w-5 text-muted-foreground transition-transform",
              !advancedOpen && "-rotate-90"
            )}
          />
          <span className="text-[15px] font-medium">Advanced Options</span>
        </button>
        {advancedOpen && (
          <div className="flex flex-col gap-2.5">
            <div className="flex items-center gap-3 rounded-[10px] border border-border bg-input px-4 py-3">
              <span className="min-w-30 text-[13px]">Weirdness</span>
              <Slider
                value={[temperature]}
                onValueChange={([v]) => setTemperature(v)}
                min={0.5}
                max={1.5}
                step={0.1}
                className="flex-1"
              />
              <span className="min-w-10 text-right text-[13px] text-muted-foreground">
                {temperature.toFixed(1)}
              </span>
            </div>

            <div className="flex items-center gap-3 rounded-[10px] border border-border bg-input px-4 py-3">
              <span className="min-w-30 text-[13px]">Style Influence</span>
              <Slider
                value={[cfgScale]}
                onValueChange={([v]) => setCfgScale(v)}
                min={1}
                max={4}
                step={0.1}
                className="flex-1"
              />
              <span className="min-w-10 text-right text-[13px] text-muted-foreground">
                {cfgScale.toFixed(1)}
              </span>
            </div>

            <div className="flex items-center gap-3 rounded-[10px] border border-border bg-input px-4 py-3">
              <span className="min-w-30 text-[13px]">Duration (seconds)</span>
              <Slider
                value={[duration]}
                onValueChange={([v]) => setDuration(v)}
                min={5}
                max={300}
                step={5}
                className="flex-1"
              />
              <span className="min-w-10 text-right text-[13px] text-muted-foreground">
                {duration}s
              </span>
            </div>

            {/* Song Title */}
            <div className="flex items-center gap-3 rounded-[10px] border border-border bg-input px-4 py-3">
              <Music className="h-4.5 w-4.5 text-muted-foreground" />
              <Input
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                placeholder="Song Title (Optional)"
                className="h-auto border-0 bg-transparent p-0 text-sm focus-visible:ring-0"
              />
            </div>
          </div>
        )}
      </section>

      <Button
        variant="outline"
        className="mt-6 w-full gap-2 py-6 text-[15px] font-medium"
        disabled={state.isGenerating}
        onClick={handleGenerate}
      >
        {state.isGenerating ? (
          <>
            <Loader2 className="h-4 w-4 animate-spin" />
            Generating...
          </>
        ) : (
          <>
            <Sparkles className="h-4 w-4" />
            Create
          </>
        )}
      </Button>

      {state.isGenerating && (
        <div className="mt-4">
          <Progress value={state.progress * 100} className="h-1" />
          <div className="mt-2 flex items-center justify-between">
            <span className="text-xs text-muted-foreground">{state.progressMessage}</span>
            <Button
              variant="outline"
              size="sm"
              className="text-xs hover:border-destructive hover:text-destructive"
              onClick={handleCancel}
            >
              Cancel
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}

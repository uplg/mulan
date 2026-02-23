import { useState, useMemo } from "react";
import { Search, SlidersHorizontal, ArrowDownWideNarrow, Play } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useApp } from "@/lib/store";
import { getAlbumArtStyle, formatDuration } from "@/lib/helpers";

type SortOrder = "newest" | "oldest" | "name";

export function WorkspacePanel() {
  const { state, dispatch } = useApp();
  const [searchQuery, setSearchQuery] = useState("");
  const [sortOrder, setSortOrder] = useState<SortOrder>("newest");

  const displaySongs = useMemo(() => {
    let filtered = state.songs;
    if (searchQuery) {
      const q = searchQuery.toLowerCase();
      filtered = filtered.filter(
        (s) =>
          s.title.toLowerCase().includes(q) ||
          s.styles.toLowerCase().includes(q)
      );
    }
    const sorted = [...filtered];
    if (sortOrder === "oldest") sorted.sort((a, b) => a.created_at - b.created_at);
    else if (sortOrder === "name") sorted.sort((a, b) => a.title.localeCompare(b.title));
    else sorted.sort((a, b) => b.created_at - a.created_at);
    return sorted;
  }, [state.songs, searchQuery, sortOrder]);

  const cycleSortOrder = () => {
    const orders: SortOrder[] = ["newest", "oldest", "name"];
    const idx = orders.indexOf(sortOrder);
    setSortOrder(orders[(idx + 1) % orders.length]);
  };

  return (
    <aside className="flex w-90 shrink-0 flex-col bg-background p-5">
      <div className="mb-5 flex items-center gap-2">
        <span className="text-sm text-muted-foreground">Workspaces</span>
        <span className="text-sm text-muted-foreground">&rsaquo; My Workspace</span>
      </div>

      <div className="mb-5 flex gap-3">
        <div className="flex flex-1 items-center gap-2 rounded-lg border border-border bg-input px-3 py-2">
          <Search className="h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="h-auto border-0 bg-transparent p-0 text-sm focus-visible:ring-0"
          />
        </div>
        <Button variant="outline" size="sm" className="gap-1.5 shrink-0">
          <SlidersHorizontal className="h-4 w-4" />
          Filters
        </Button>
        <Button
          variant="outline"
          size="sm"
          className="gap-1.5 shrink-0"
          onClick={cycleSortOrder}
        >
          <ArrowDownWideNarrow className="h-4 w-4" />
          {sortOrder.charAt(0).toUpperCase() + sortOrder.slice(1)}
        </Button>
      </div>

      <ScrollArea className="flex-1">
        {displaySongs.length === 0 ? (
          <div className="py-16 text-center text-sm text-muted-foreground">
            {searchQuery ? "No songs match your search" : "Your generated songs will appear here"}
          </div>
        ) : (
          <div className="flex flex-col gap-2">
            {displaySongs.map((song) => (
              <button
                key={song.filename}
                className="flex items-center gap-3 rounded-lg p-3 text-left transition-colors hover:bg-accent group"
                onClick={() => dispatch({ type: "PLAY_TRACK", track: song })}
              >
                <div
                  className="relative h-14 w-14 shrink-0 overflow-hidden rounded-md"
                  style={getAlbumArtStyle(song)}
                >
                  <span className="absolute bottom-1 right-1 rounded bg-black/75 px-1.5 py-0.5 text-[10px] text-white">
                    {formatDuration(song.duration)}
                  </span>
                </div>
                <div className="min-w-0 flex-1">
                  <h4 className="truncate text-sm font-medium text-foreground">
                    {song.title}
                  </h4>
                  <p className="truncate text-xs text-muted-foreground">
                    {song.styles || "No styles"}
                  </p>
                </div>
                <div className="opacity-0 group-hover:opacity-100 transition-opacity">
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8"
                    onClick={(e) => {
                      e.stopPropagation();
                      dispatch({ type: "PLAY_TRACK", track: song });
                    }}
                  >
                    <Play className="h-4 w-4" />
                  </Button>
                </div>
              </button>
            ))}
          </div>
        )}
      </ScrollArea>
    </aside>
  );
}

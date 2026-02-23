import { useEffect } from "react";
import { TooltipProvider } from "@/components/ui/tooltip";
import { AppProvider, useApp } from "@/lib/store";
import { api } from "@/lib/api";
import { Sidebar } from "@/components/layout/Sidebar";
import { WorkspacePanel } from "@/components/workspace/WorkspacePanel";
import { CreateView } from "@/components/create/CreateView";
import { PlayerBar } from "@/components/player/PlayerBar";

function AppContent() {
  const { state, dispatch } = useApp();

  // Load songs on mount
  useEffect(() => {
    api.getSongs().then((data) => {
      dispatch({ type: "SET_SONGS", songs: data.songs });
    }).catch(console.error);
  }, [dispatch]);

  // Check if generation is already in progress on mount
  useEffect(() => {
    api.getProgress().then((status) => {
      if (status.is_generating) {
        dispatch({ type: "SET_GENERATING", isGenerating: true });
        dispatch({ type: "SET_PROGRESS", progress: status.progress, message: status.message });
      }
    }).catch(() => {});
  }, [dispatch]);

  return (
    <div className="flex h-screen w-full flex-col">
      <div className="flex flex-1 overflow-hidden">
        <Sidebar />

        {/* Main content area */}
        <main className="flex-1 overflow-y-auto border-r border-border px-8 py-6">
          {state.view === "create" && <CreateView />}
          {state.view === "home" && (
            <div>
              <h1 className="mb-2 text-[28px] font-bold">Welcome to HeartMuLa</h1>
              <p className="text-sm text-muted-foreground">AI Music Generation powered by MLX</p>
            </div>
          )}
          {state.view === "library" && (
            <div>
              <h1 className="mb-2 text-[28px] font-bold">Library</h1>
              <p className="text-sm text-muted-foreground">All generated songs from outputs folder</p>
            </div>
          )}
          {state.view === "studio" && (
            <div>
              <h1 className="mb-2 text-[28px] font-bold">Studio</h1>
              <p className="text-sm text-muted-foreground">Your music generation workspace</p>
            </div>
          )}
          {state.view === "search" && (
            <div>
              <h1 className="mb-2 text-[28px] font-bold">Search</h1>
              <p className="text-sm text-muted-foreground">Find songs by title, styles, or lyrics</p>
            </div>
          )}
          {state.view === "settings" && (
            <div>
              <h1 className="mb-2 text-[28px] font-bold">Settings</h1>
              <p className="text-sm text-muted-foreground">Configure HeartMuLa</p>
            </div>
          )}
        </main>

        {/* Workspace panel (visible on create view) */}
        {state.view === "create" && <WorkspacePanel />}
      </div>

      {/* Player bar */}
      <PlayerBar />

      {/* Add bottom padding when player is visible */}
      {state.currentTrack && <div className="h-[90px] shrink-0" />}
    </div>
  );
}

export default function App() {
  return (
    <AppProvider>
      <TooltipProvider>
        <AppContent />
      </TooltipProvider>
    </AppProvider>
  );
}

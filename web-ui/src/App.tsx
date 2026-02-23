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

  useEffect(() => {
    api.getSongs().then((data) => {
      dispatch({ type: "SET_SONGS", songs: data.songs });
    }).catch(console.error);
  }, [dispatch]);

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

        <main className="flex-1 overflow-y-auto border-r border-border px-8 py-6">
          {state.view === "create" && <CreateView />}
        </main>

        {state.view === "create" && <WorkspacePanel />}
      </div>

      <PlayerBar />

      {(state.currentTrack || state.isStreaming) && <div className="h-22.5 shrink-0" />}
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

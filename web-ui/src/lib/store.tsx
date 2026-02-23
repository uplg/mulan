import { createContext, useContext, useReducer, type ReactNode } from "react";
import type { SongInfo } from "@/lib/types";

// ----- State -----
export interface AppState {
  songs: SongInfo[];
  isGenerating: boolean;
  progress: number;
  progressMessage: string;
  currentTrack: SongInfo | null;
  isPlaying: boolean;
  view: "create" | "home" | "library" | "search" | "studio" | "settings";
  /** True while streaming audio segments are being played */
  isStreaming: boolean;
  /** When set, PlayerBar should seek to this position (seconds) after loading the track */
  seekTo: number | null;
}

const initialState: AppState = {
  songs: [],
  isGenerating: false,
  progress: 0,
  progressMessage: "Ready",
  currentTrack: null,
  isPlaying: false,
  view: "create",
  isStreaming: false,
  seekTo: null,
};

// ----- Actions -----
type Action =
  | { type: "SET_SONGS"; songs: SongInfo[] }
  | { type: "SET_GENERATING"; isGenerating: boolean }
  | { type: "SET_PROGRESS"; progress: number; message: string }
  | { type: "PLAY_TRACK"; track: SongInfo; seekTo?: number }
  | { type: "SET_PLAYING"; isPlaying: boolean }
  | { type: "SET_VIEW"; view: AppState["view"] }
  | { type: "SET_STREAMING"; isStreaming: boolean }
  | { type: "CLEAR_SEEK" };

function reducer(state: AppState, action: Action): AppState {
  switch (action.type) {
    case "SET_SONGS":
      return { ...state, songs: action.songs };
    case "SET_GENERATING":
      return { ...state, isGenerating: action.isGenerating };
    case "SET_PROGRESS":
      return { ...state, progress: action.progress, progressMessage: action.message };
    case "PLAY_TRACK":
      return { ...state, currentTrack: action.track, isPlaying: true, isStreaming: false, seekTo: action.seekTo ?? null };
    case "SET_PLAYING":
      return { ...state, isPlaying: action.isPlaying };
    case "SET_VIEW":
      return { ...state, view: action.view };
    case "SET_STREAMING":
      return { ...state, isStreaming: action.isStreaming };
    case "CLEAR_SEEK":
      return { ...state, seekTo: null };
    default:
      return state;
  }
}

// ----- Context -----
const AppContext = createContext<{
  state: AppState;
  dispatch: React.Dispatch<Action>;
} | null>(null);

export function AppProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(reducer, initialState);
  return (
    <AppContext.Provider value={{ state, dispatch }}>
      {children}
    </AppContext.Provider>
  );
}

export function useApp() {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error("useApp must be used within AppProvider");
  return ctx;
}

// API types matching the FastAPI server models

export interface GenerateRequest {
  lyrics: string;
  styles: string;
  duration: number;
  cfg_scale: number;
  temperature: number;
  topk: number;
  title: string;
  language: string;
}

export interface GenerateResponse {
  filename: string;
  frames: number;
  duration: number;
  time: number;
}

export interface ProgressResponse {
  progress: number;
  message: string;
  is_generating: boolean;
  current_request: {
    title: string;
    styles: string;
    lyrics: string;
    duration: number;
  } | null;
  started_at: number | null;
}

export interface SongInfo {
  filename: string;
  title: string;
  styles: string;
  lyrics: string;
  duration: number;
  created_at: number;
  album_art: string | null;
}

export interface SongsResponse {
  songs: SongInfo[];
}

export interface SurpriseLyrics {
  title: string;
  styles: string;
  lyrics: string;
}

export interface StatusResponse {
  status: string;
  model_loaded: boolean;
  is_generating: boolean;
}

// SSE streaming event types

export interface StreamProgressEvent {
  progress: number;
  message: string;
}

export interface StreamCompleteEvent {
  filename: string;
  frames: number;
  duration: number;
  time: number;
}

export interface StreamCancelledEvent {
  message: string;
}

import type {
  GenerateRequest,
  GenerateResponse,
  ProgressResponse,
  SongsResponse,
  SurpriseLyrics,
  StatusResponse,
} from "./types";

const API_BASE = "/api";

async function fetchJSON<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${url}`, init);
  if (!res.ok) {
    throw new Error(`API error ${res.status}: ${res.statusText}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  getStatus: () => fetchJSON<StatusResponse>("/status"),

  getProgress: (signal?: AbortSignal) =>
    fetchJSON<ProgressResponse>("/progress", { signal }),

  getSongs: () => fetchJSON<SongsResponse>("/songs"),

  getSurpriseLyrics: () => fetchJSON<SurpriseLyrics>("/surprise-lyrics"),

  generate: (request: GenerateRequest, signal?: AbortSignal) =>
    fetchJSON<GenerateResponse>("/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
      signal,
    }),

  cancel: () =>
    fetchJSON<{ status: string; message: string }>("/cancel", {
      method: "POST",
    }),

  audioUrl: (filename: string) => `${API_BASE}/audio/${filename}`,
};

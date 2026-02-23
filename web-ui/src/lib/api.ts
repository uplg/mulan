import type {
  GenerateRequest,
  GenerateResponse,
  ProgressResponse,
  SongsResponse,
  SurpriseLyrics,
  StatusResponse,
  StreamProgressEvent,
  StreamCompleteEvent,
  StreamCancelledEvent,
} from "./types";

const API_BASE = "/api";

async function fetchJSON<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${url}`, init);
  if (!res.ok) {
    throw new Error(`API error ${res.status}: ${res.statusText}`);
  }
  return res.json() as Promise<T>;
}

// Binary framing protocol constants (must match server.py)
const FRAME_AUDIO = 0x01;
const FRAME_PROGRESS = 0x02;
const FRAME_COMPLETE = 0x03;
const FRAME_CANCELLED = 0x04;
const FRAME_HEADER_SIZE = 5; // 1 byte type + 4 bytes length

export interface AudioChunkInfo {
  /** Raw audio bytes — no base64 overhead */
  bytes: Uint8Array;
  format: "mp3" | "wav";
  segment: number;
  totalSegments: number;
}

export interface StreamCallbacks {
  onAudio: (chunk: AudioChunkInfo) => void;
  onProgress: (event: StreamProgressEvent) => void;
  onComplete: (event: StreamCompleteEvent) => void;
  onCancelled: (event: StreamCancelledEvent) => void;
  onError: (error: Error) => void;
}

/**
 * Read exactly `n` bytes from a ReadableStream reader, buffering across chunks.
 * Returns null if the stream ends before `n` bytes are available.
 */
async function readExact(
  reader: ReadableStreamDefaultReader<Uint8Array>,
  n: number,
  leftover: { buf: Uint8Array; offset: number },
): Promise<Uint8Array | null> {
  const result = new Uint8Array(n);
  let filled = 0;

  // Drain leftover buffer first
  if (leftover.offset < leftover.buf.length) {
    const available = leftover.buf.length - leftover.offset;
    const take = Math.min(available, n);
    result.set(leftover.buf.subarray(leftover.offset, leftover.offset + take), 0);
    leftover.offset += take;
    filled += take;
  }

  while (filled < n) {
    const { done, value } = await reader.read();
    if (done || !value) return null;

    const need = n - filled;
    if (value.length <= need) {
      result.set(value, filled);
      filled += value.length;
    } else {
      result.set(value.subarray(0, need), filled);
      filled += need;
      // Store the remainder in leftover
      leftover.buf = value;
      leftover.offset = need;
    }
  }

  return result;
}

/** Parse the binary stream from /api/generate-stream and dispatch callbacks. */
async function consumeBinaryStream(
  response: Response,
  callbacks: StreamCallbacks,
): Promise<void> {
  const reader = response.body?.getReader();
  if (!reader) {
    callbacks.onError(new Error("No response body"));
    return;
  }

  const leftover = { buf: new Uint8Array(0), offset: 0 };
  const decoder = new TextDecoder();

  try {
    while (true) {
      // Read frame header: 1 byte type + 4 bytes big-endian length
      const header = await readExact(reader, FRAME_HEADER_SIZE, leftover);
      if (!header) break; // stream ended

      const frameType = header[0];
      const payloadLen =
        (header[1] << 24) | (header[2] << 16) | (header[3] << 8) | header[4];

      // Read payload
      const payload = await readExact(reader, payloadLen, leftover);
      if (!payload) break;

      switch (frameType) {
        case FRAME_AUDIO: {
          // First 4 bytes: [format, segment, totalSegments, reserved]
          const format = payload[0] === 0 ? "mp3" : "wav";
          const segment = payload[1];
          const totalSegments = payload[2];
          const audioBytes = payload.subarray(4);
          callbacks.onAudio({
            bytes: audioBytes,
            format: format as "mp3" | "wav",
            segment,
            totalSegments,
          });
          break;
        }
        case FRAME_PROGRESS: {
          const data = JSON.parse(decoder.decode(payload)) as StreamProgressEvent;
          callbacks.onProgress(data);
          break;
        }
        case FRAME_COMPLETE: {
          const data = JSON.parse(decoder.decode(payload)) as StreamCompleteEvent;
          callbacks.onComplete(data);
          break;
        }
        case FRAME_CANCELLED: {
          const data = JSON.parse(decoder.decode(payload)) as StreamCancelledEvent;
          callbacks.onCancelled(data);
          break;
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
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

  generateStream: async (
    request: GenerateRequest,
    callbacks: StreamCallbacks,
    signal?: AbortSignal,
  ): Promise<void> => {
    const res = await fetch(`${API_BASE}/generate-stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
      signal,
    });
    if (!res.ok) {
      throw new Error(`API error ${res.status}: ${res.statusText}`);
    }
    await consumeBinaryStream(res, callbacks);
  },

  cancel: () =>
    fetchJSON<{ status: string; message: string }>("/cancel", {
      method: "POST",
    }),

  audioUrl: (filename: string) => `${API_BASE}/audio/${filename}`,
};

"use client";

import { usePathname, useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { useIdentify } from "./Identify";

const IDLE = "Tap microphone to identify music";

function pickMimeType(): string {
  if (typeof MediaRecorder === "undefined") return "";
  for (const t of ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg;codecs=opus"]) {
    if (MediaRecorder.isTypeSupported(t)) return t;
  }
  return "";
}

// Record or hum: up to 12 s from the microphone, sent to /api/identify. The result shows on the home page.
export default function Recorder() {
  const { run } = useIdentify();
  const router = useRouter();
  const pathname = usePathname();
  const [recording, setRecording] = useState(false);
  const [status, setStatus] = useState(IDLE);
  const recorder = useRef<MediaRecorder | null>(null);
  const stream = useRef<MediaStream | null>(null);
  const audioCtx = useRef<AudioContext | null>(null);
  const analyser = useRef<AnalyserNode | null>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const autoStop = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Frequency bars while recording
  useEffect(() => {
    const cv = canvas.current;
    const an = analyser.current;
    if (!recording || !cv || !an) return;
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    const data = new Uint8Array(an.frequencyBinCount);
    let animId = 0;
    const draw = () => {
      animId = requestAnimationFrame(draw);
      an.getByteFrequencyData(data);
      ctx.clearRect(0, 0, cv.width, cv.height);
      const barWidth = (cv.width / data.length) * 1.5;
      let x = 0;
      for (let i = 0; i < data.length; i++) {
        const barHeight = (data[i] / 255) * cv.height;
        ctx.fillStyle = `rgba(139, 92, 246, ${data[i] / 255 + 0.2})`;
        ctx.fillRect(x, cv.height - barHeight, barWidth, barHeight);
        x += barWidth + 2;
      }
    };
    draw();
    return () => cancelAnimationFrame(animId);
  }, [recording]);

  // Leaving the page mid-recording releases the microphone
  useEffect(() => () => {
    if (autoStop.current) clearTimeout(autoStop.current);
    if (recorder.current) recorder.current.onstop = null;
    stream.current?.getTracks().forEach((t) => t.stop());
    audioCtx.current?.close().catch(() => {});
  }, []);

  const start = (s: MediaStream) => {
    const chunks: Blob[] = [];
    const mimeType = pickMimeType();
    const mr = mimeType ? new MediaRecorder(s, { mimeType }) : new MediaRecorder(s);
    recorder.current = mr;
    stream.current = s;

    const Ctx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
    audioCtx.current = new Ctx();
    const an = audioCtx.current.createAnalyser();
    an.fftSize = 64;
    audioCtx.current.createMediaStreamSource(s).connect(an);
    analyser.current = an;

    mr.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) chunks.push(e.data);
    };
    mr.onstop = () => {
      if (autoStop.current) clearTimeout(autoStop.current);
      s.getTracks().forEach((t) => t.stop());
      audioCtx.current?.close().catch(() => {});
      setRecording(false);
      setStatus("Analyzing audio with Echo neural encoder...");
      const blob = new Blob(chunks, { type: mr.mimeType || "audio/webm" });
      if (pathname !== "/") router.push("/");
      run(blob).finally(() => setStatus(IDLE));
    };

    mr.start(250);
    setRecording(true);
    setStatus("Listening... Tap again to identify");
    autoStop.current = setTimeout(() => {
      if (mr.state === "recording") mr.stop();
    }, 12000);
  };

  const onClick = async () => {
    if (recorder.current && recorder.current.state === "recording") {
      recorder.current.stop();
      return;
    }
    try {
      start(await navigator.mediaDevices.getUserMedia({ audio: true }));
    } catch (err) {
      alert("Microphone access denied or unavailable: " + (err as Error).message);
    }
  };

  return (
    <div className="record-container">
      <button className={`record-btn${recording ? " recording" : ""}`} onClick={onClick}
        title="Click to record audio from microphone" aria-label={recording ? "Stop recording" : "Record audio"}>
        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
          aria-hidden="true">
          <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
          <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
          <line x1="12" y1="19" x2="12" y2="23" />
          <line x1="8" y1="23" x2="16" y2="23" />
        </svg>
      </button>
      {recording && <canvas ref={canvas} className="visualizer" />}
      <div className="record-status">{status}</div>
    </div>
  );
}

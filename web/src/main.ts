import { PipecatClient, type RTVIMessage } from "@pipecat-ai/client-js";
import { SmallWebRTCTransport } from "@pipecat-ai/small-webrtc-transport";
import "./style.css";
import {
  deleteConversation,
  getConversation,
  listConversations,
  restartLlm,
} from "./api";
import { Orb, type OrbState } from "./orb";
import { Sidebar } from "./sidebar";
import { Transcript } from "./transcript";

type Conn = "idle" | "connecting" | "live" | "ended" | "error";
type Phase = "listening" | "userSpeaking" | "thinking" | "speaking";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const orb = new Orb($("orb") as HTMLCanvasElement);
const transcript = new Transcript($("transcript"));
const captionEl = $("caption");
const noticeEl = $("notice");
const actionsEl = $("actions");
const muteBtn = $<HTMLButtonElement>("mute-btn");
const stopBtn = $<HTMLButtonElement>("stop-btn");

let client: PipecatClient | null = null;
let conn: Conn = "idle";
let phase: Phase = "listening";
let muted = false;
let pttHeld = false;
let activeId: string | null = null;
let generation = 0;
const botAudio = $<HTMLAudioElement>("bot-audio");
let audioCtx: AudioContext | null = null;
let analyser: AnalyserNode | null = null;
let analyserBuf: Uint8Array<ArrayBuffer> | null = null;
let analyserSrc: MediaStreamAudioSourceNode | null = null;
let botLevel = 0;
let levelRaf = 0;
let thinkingTimer: number | undefined;
const THINKING_TIMEOUT_MS = 8000;

const sidebar = new Sidebar($("conv-list"), $("conv-empty"), {
  onSelect: (id) => void connect(id),
  onDelete: (id) => void removeConversation(id),
});

// ---------- rendering ----------

function orbState(): OrbState {
  switch (conn) {
    case "connecting":
      return "connecting";
    case "error":
    case "ended":
      return "error";
    case "idle":
      return "muted";
    case "live":
      if ((muted && !pttHeld) && (phase === "listening" || phase === "userSpeaking")) return "muted";
      return phase;
  }
}

const CAPTIONS: Record<OrbState, string> = {
  connecting: "Connecting…",
  listening: "Listening",
  userSpeaking: "Hearing you",
  thinking: "Thinking…",
  speaking: "Speaking",
  muted: "Muted — hold Space to talk",
  error: "",
};

function render(): void {
  const s = orbState();
  orb.setState(s);
  if (s !== "userSpeaking" && s !== "speaking" && s !== "listening") orb.setLevel(0);
  if (conn === "idle") captionEl.textContent = "Not connected";
  else if (conn === "ended") captionEl.textContent = "Session ended";
  else if (conn === "error") captionEl.textContent = "Something went wrong";
  else captionEl.textContent = CAPTIONS[s];
  muteBtn.disabled = conn !== "live";
  muteBtn.textContent = muted ? "Unmute" : "Mute";
  muteBtn.setAttribute("aria-pressed", String(muted));
  stopBtn.disabled = conn !== "live" && conn !== "connecting";
  renderActions();
}

function button(label: string, onClick: () => void, primary = false): HTMLButtonElement {
  const b = document.createElement("button");
  b.type = "button";
  b.className = "btn" + (primary ? " primary" : "");
  b.textContent = label;
  b.addEventListener("click", onClick);
  return b;
}

function renderActions(): void {
  actionsEl.replaceChildren();
  if (conn === "idle") {
    actionsEl.append(button("Start new conversation", () => void connect(null), true));
    const last = sidebar.newest();
    if (last) actionsEl.append(button("Continue last conversation", () => void connect(last.id)));
  } else if (conn === "ended") {
    actionsEl.append(button("Reconnect", () => void connect(activeId), true));
  } else if (conn === "error") {
    actionsEl.append(button("Reconnect", () => void connect(activeId), true));
    actionsEl.append(button("Restart LLM", () => void restartAndReconnect()));
  }
}

function showNotice(content: string | Node | null): void {
  noticeEl.replaceChildren();
  if (content === null) {
    noticeEl.hidden = true;
    return;
  }
  noticeEl.append(content);
  noticeEl.hidden = false;
}

function micDeniedNotice(): HTMLElement {
  const wrap = document.createElement("div");
  const p = document.createElement("strong");
  p.textContent = "Microphone access was blocked.";
  const ul = document.createElement("ul");
  for (const t of [
    "Chrome: click the lock icon in the address bar → Site settings → Microphone → Allow, then Reconnect.",
    "Safari: Safari menu → Settings → Websites → Microphone → set this site to Allow, then Reconnect.",
  ]) {
    const li = document.createElement("li");
    li.textContent = t;
    ul.append(li);
  }
  wrap.append(p, ul);
  return wrap;
}

function isMicDenied(e: unknown): boolean {
  const name = (e as { name?: string })?.name ?? "";
  const msg = String((e as { message?: string })?.message ?? e);
  return (
    name === "NotAllowedError" ||
    name === "PermissionDeniedError" ||
    /(microphone|getUserMedia).*(denied|not allowed|permission)|permission.*(microphone|getUserMedia)/i.test(msg)
  );
}

// client-js/transport swallow getUserMedia failures (connect() still resolves),
// so mic problems arrive via onDeviceError / mediaState instead of a rejection.
function micProblemNotice(kind: string, detail: string): string | HTMLElement {
  if (kind === "permissions" || kind === "blocked") return micDeniedNotice();
  return `Microphone problem (${kind}): ${detail}`;
}

// ---------- connection ----------

async function refreshSidebar(): Promise<void> {
  try {
    sidebar.setConversations(await listConversations());
  } catch (e) {
    console.warn("could not list conversations", e);
  }
}

function stopBotAudio(): void {
  cancelAnimationFrame(levelRaf);
  levelRaf = 0;
  botLevel = 0;
  botAudio.srcObject = null;
  try {
    analyserSrc?.disconnect();
  } catch {
    /* ignore */
  }
  analyserSrc = null;
  analyser = null;
  analyserBuf = null;
  void audioCtx?.close().catch(() => {});
  audioCtx = null;
}

function levelLoop(): void {
  levelRaf = requestAnimationFrame(levelLoop);
  if (!analyser || !analyserBuf) return;
  analyser.getByteTimeDomainData(analyserBuf);
  let sum = 0;
  for (const v of analyserBuf) {
    const d = (v - 128) / 128;
    sum += d * d;
  }
  botLevel = Math.sqrt(sum / analyserBuf.length);
  if (phase === "speaking") orb.setLevel(Math.min(1, botLevel * 4));
}

function attachBotAudio(track: MediaStreamTrack): void {
  const stream = new MediaStream([track]);
  botAudio.srcObject = stream;
  botAudio.play().catch((e) => console.warn("bot audio play() rejected", e));
  // The transport's own player is disabled, so onRemoteAudioLevel never fires:
  // measure the remote stream ourselves.
  try {
    audioCtx ??= new AudioContext();
    void audioCtx.resume();
    analyserSrc?.disconnect();
    analyserSrc = audioCtx.createMediaStreamSource(stream);
    analyser = audioCtx.createAnalyser();
    analyser.fftSize = 1024;
    analyserBuf = new Uint8Array(new ArrayBuffer(analyser.fftSize));
    analyserSrc.connect(analyser); // not connected to destination: <audio> plays it
    if (!levelRaf) levelLoop();
  } catch (e) {
    console.warn("bot level analyser unavailable", e);
  }
}

async function teardown(): Promise<void> {
  stopBotAudio();
  const c = client;
  client = null;
  if (c) {
    try {
      await c.disconnect();
    } catch {
      /* already gone */
    }
  }
}

async function connect(id: string | null): Promise<void> {
  const gen = ++generation;
  conn = "idle"; // makes the old client's onDisconnected a no-op
  await teardown();
  if (gen !== generation) return;
  showNotice(null);
  muted = false;
  pttHeld = false;
  phase = "listening";
  clearTimeout(thinkingTimer);
  activeId = id;
  sidebar.setActive(id);
  if (id) {
    try {
      transcript.showHistory(await getConversation(id));
    } catch {
      transcript.clear();
    }
  } else {
    transcript.clear();
  }
  if (gen !== generation) return;
  conn = "connecting";
  render();

  let micProblem: string | HTMLElement | null = null;
  const failMic = (g: number) => {
    if (g !== generation) return;
    conn = "error";
    showNotice(micProblem);
    render();
    void teardown();
  };
  const c = new PipecatClient({
    transport: new SmallWebRTCTransport(),
    enableMic: true,
    enableCam: false,
    callbacks: {
      onDisconnected: () => {
        if (gen !== generation || conn !== "live") return;
        conn = "ended";
        showNotice("Session ended (opened in another tab or server stopped). Reconnect to continue.");
        render();
      },
      onError: (msg: RTVIMessage) => {
        if (gen !== generation || conn === "idle") return;
        console.error("RTVI error", msg);
        conn = "error";
        showNotice(`Assistant error: ${describe(msg)}`);
        render();
      },
      onDeviceError: (err) => {
        if (gen !== generation) return;
        micProblem = micProblemNotice(err.type, err.message);
        if (conn === "live") failMic(gen);
      },
      onMediaStateChanged: (ms) => {
        if (gen !== generation || ms.mic.state !== "error") return;
        micProblem ??= micProblemNotice(ms.mic.reason, String(ms.mic.details ?? ""));
        if (conn === "live") failMic(gen);
      },
      onTrackStarted: (track, participant) => {
        if (gen === generation && !participant?.local && track.kind === "audio") attachBotAudio(track);
      },
      onUserStartedSpeaking: () => setPhase(gen, "userSpeaking"),
      // Ignore out-of-order events: only leave the phase the event belongs to.
      onUserStoppedSpeaking: () => {
        if (phase === "userSpeaking") setPhase(gen, "thinking");
      },
      onBotStartedSpeaking: () => {
        transcript.startBotTurn();
        setPhase(gen, "speaking");
      },
      onBotStoppedSpeaking: () => {
        if (phase === "speaking") setPhase(gen, "listening");
      },
      onLocalAudioLevel: (l: number) => {
        if (gen === generation && (phase === "listening" || phase === "userSpeaking") && !muted) {
          orb.setLevel(Math.min(1, l * 2));
        }
      },
      onRemoteAudioLevel: (l: number) => {
        if (gen === generation && phase === "speaking") orb.setLevel(Math.min(1, l * 2));
      },
      onUserTranscript: (d) => {
        if (gen === generation && d.final && d.text.trim()) transcript.addUser(d.text.trim());
      },
      onBotTtsText: (d) => {
        if (gen === generation && d.text.trim()) transcript.appendBot(d.text);
      },
    },
  });
  client = c;

  try {
    await c.connect({
      webrtcRequestParams: {
        endpoint: "/api/offer" + (id ? "?conversation_id=" + encodeURIComponent(id) : ""),
      },
    });
  } catch (e) {
    if (gen !== generation) return;
    console.error("connect failed", e);
    conn = "error";
    showNotice(isMicDenied(e) ? micDeniedNotice() : `Could not connect: ${describe(e)}`);
    render();
    return;
  }
  if (gen !== generation) return;
  if ((conn as Conn) === "error") return; // onError fired during connect; keep the error state
  if (!micProblem && c.mediaState.mic.state === "error") {
    const m = c.mediaState.mic;
    micProblem = micProblemNotice(m.reason, String(m.details ?? ""));
  }
  if (micProblem) {
    failMic(gen);
    return;
  }
  conn = "live";
  phase = "listening";
  render();

  // X-Conversation-Id is not readable via client-js: pick the newest one.
  try {
    const list = await listConversations();
    if (gen !== generation) return;
    sidebar.setConversations(list);
    if ((!id || !list.some((x) => x.id === id)) && list[0]) {
      activeId = list[0].id;
      sidebar.setActive(activeId);
    }
  } catch (e) {
    console.warn("could not list conversations", e);
  }
}

function describe(e: unknown): string {
  if (e instanceof Error) return e.message;
  const m = e as { data?: { message?: string; error?: string } };
  return m?.data?.message ?? m?.data?.error ?? "unknown error";
}

function setPhase(gen: number, p: Phase): void {
  if (gen !== generation || conn !== "live") return;
  phase = p;
  clearTimeout(thinkingTimer);
  if (p === "thinking") {
    // VAD can fire while STT returns nothing; don't swirl forever.
    thinkingTimer = window.setTimeout(() => {
      if (gen === generation && phase === "thinking") setPhase(gen, "listening");
    }, THINKING_TIMEOUT_MS);
  }
  orb.setLevel(0);
  render();
}

async function disconnectByUser(): Promise<void> {
  generation++;
  conn = "idle";
  showNotice(null);
  await teardown();
  render();
}

async function restartAndReconnect(): Promise<void> {
  if (conn !== "error") return; // already in flight
  const id = activeId;
  generation++;
  conn = "connecting"; // render() below removes the Reconnect/Restart buttons
  showNotice(null);
  render();
  captionEl.textContent = "Restarting LLM…";
  await teardown();
  try {
    await restartLlm();
  } catch (e) {
    conn = "error";
    showNotice(`LLM restart failed: ${describe(e)}`);
    render();
    return;
  }
  await connect(id);
}

async function removeConversation(id: string): Promise<void> {
  try {
    await deleteConversation(id);
  } catch (e) {
    showNotice(`Could not delete: ${describe(e)}`);
    return;
  }
  if (id === activeId) {
    await disconnectByUser();
    activeId = null;
    sidebar.setActive(null);
    transcript.clear();
  }
  await refreshSidebar();
  render();
}

// ---------- controls ----------

function setMic(on: boolean): void {
  try {
    client?.enableMic(on);
  } catch (e) {
    console.warn("enableMic failed", e);
  }
}

muteBtn.addEventListener("click", () => {
  if (conn !== "live") return;
  muteBtn.blur(); // keep Space from re-clicking the button
  muted = !muted;
  pttHeld = false; // an in-flight Space hold must not re-mute on keyup
  setMic(!muted);
  render();
});
stopBtn.addEventListener("click", () => void disconnectByUser());
$("new-btn").addEventListener("click", () => void connect(null));

function inTextField(t: EventTarget | null): boolean {
  return (
    t instanceof HTMLInputElement ||
    t instanceof HTMLTextAreaElement ||
    t instanceof HTMLSelectElement ||
    (t instanceof HTMLElement && t.isContentEditable)
  );
}

function releasePtt(): void {
  if (!pttHeld) return;
  pttHeld = false;
  setMic(false);
  render();
}

window.addEventListener("keydown", (e) => {
  if (inTextField(e.target)) return;
  if (e.key === "Escape") {
    if (conn === "live") {
      try {
        client?.sendClientMessage("interrupt");
      } catch (err) {
        console.warn("interrupt failed", err);
      }
    }
    return;
  }
  if (e.code === "Space" && conn === "live" && muted) {
    e.preventDefault(); // also stops a focused button from "clicking"
    if (e.repeat || pttHeld) return;
    pttHeld = true;
    setMic(true);
    render();
  }
});
window.addEventListener("keyup", (e) => {
  if (e.code === "Space" && pttHeld) {
    e.preventDefault();
    releasePtt();
  }
});
window.addEventListener("blur", releasePtt);

// ---------- boot ----------

// Exposed for automated checks only.
(window as unknown as { __voice: () => unknown }).__voice = () => ({
  conn,
  orb: orb.getState(),
  activeId,
  muted,
  botLevel,
  audioPaused: botAudio.paused,
  audioTrack: (botAudio.srcObject as MediaStream | null)?.getAudioTracks().map((t) => t.readyState) ?? null,
  ctx: audioCtx?.state ?? null,
});

void refreshSidebar().then(render);
render();

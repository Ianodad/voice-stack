import { PipecatClient, type RTVIMessage } from "@pipecat-ai/client-js";
import { SmallWebRTCTransport } from "@pipecat-ai/small-webrtc-transport";
import "./style.css";
import { ActionCard, RESULT_TEXT, type PublicAction } from "./actions";
import {
  ApiError,
  approveAction,
  denyAction,
  pendingActions,
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
const activityEl = $("activity");
const card = new ActionCard($("action-card-host"), $("action-result"));

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

const MIC_STOPPED = "Microphone stopped (device changed or in use) \u2014 click Reconnect";

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

const soundBanner = $<HTMLButtonElement>("sound-banner");
let botTrack: MediaStreamTrack | null = null;
let analyserTrack: MediaStreamTrack | null = null;
let soundCheckTimer: number | undefined;
let localTrack: MediaStreamTrack | null = null;

/** Log the full playback state so a silent browser can be diagnosed from the console. */
function logAudio(tag: string): { blocked: boolean } {
  const s = botAudio.srcObject as MediaStream | null;
  const info = {
    tag,
    readyState: botAudio.readyState,
    paused: botAudio.paused,
    muted: botAudio.muted,
    volume: botAudio.volume,
    sinkId: (botAudio as HTMLAudioElement & { sinkId?: string }).sinkId ?? "n/a",
    tracks: s?.getAudioTracks().map((t) => ({ enabled: t.enabled, muted: t.muted, readyState: t.readyState })) ?? null,
    audioContext: audioCtx?.state ?? null,
  };
  console.log("[bot-audio]", JSON.stringify(info));
  return { blocked: !!s && (botAudio.paused || botAudio.muted || botAudio.volume === 0) };
}

function checkSound(tag: string): void {
  const { blocked } = logAudio(tag);
  soundBanner.hidden = !(blocked && botAudio.srcObject);
}

function scheduleSoundCheck(tag: string, delay = 1200): void {
  clearTimeout(soundCheckTimer);
  soundCheckTimer = window.setTimeout(() => checkSound(tag), delay);
}

for (const ev of ["play", "pause", "volumechange", "ended", "stalled", "error"]) {
  botAudio.addEventListener(ev, () => {
    if (botAudio.srcObject) scheduleSoundCheck(`audio:${ev}`, 200);
  });
}

soundBanner.addEventListener("click", () => {
  botAudio.muted = false;
  if (botAudio.volume === 0) botAudio.volume = 1;
  void audioCtx?.resume().catch(() => {});
  botAudio
    .play()
    .catch((e) => console.warn("bot audio play() rejected after click", e))
    .finally(() => scheduleSoundCheck("banner-click", 300));
});

function stopBotAudio(): void {
  clearTimeout(soundCheckTimer);
  soundBanner.hidden = true;
  botTrack = null;
  analyserTrack?.stop();
  analyserTrack = null;
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
  botTrack = track;
  const stream = new MediaStream([track]);
  botAudio.srcObject = stream;
  botAudio.muted = false;
  for (const ev of ["mute", "unmute", "ended"] as const) {
    track.addEventListener(ev, () => {
      if (track === botTrack) scheduleSoundCheck(`track:${ev}`, 200);
    });
  }
  logAudio("attach");
  botAudio
    .play()
    .then(() => scheduleSoundCheck("play-resolved"))
    .catch((e) => {
      console.warn("bot audio play() rejected", e);
      checkSound("play-rejected");
      soundBanner.hidden = false;
    });
  // The transport's own player is disabled, so onRemoteAudioLevel never fires:
  // measure the remote stream ourselves. Purely optional and isolated: it uses a
  // cloned track, is never connected to the destination, and any failure only
  // costs the orb its level animation, never playback.
  try {
    audioCtx ??= new AudioContext();
    audioCtx.onstatechange = () => logAudio("audioctx-state");
    void audioCtx.resume().catch(() => {});
    analyserSrc?.disconnect();
    analyserTrack?.stop();
    analyserTrack = track.clone();
    analyserSrc = audioCtx.createMediaStreamSource(new MediaStream([analyserTrack]));
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
  card.clear();
  clearActivity();
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
  const watchMic = (g: number, track: MediaStreamTrack) => {
    localTrack = track;
    track.addEventListener("ended", () => {
      // The transport swaps the track when the default device changes; only
      // report loss if no replacement local track appeared shortly after.
      window.setTimeout(() => {
        if (g === generation && localTrack === track && conn === "live") {
          micProblem = MIC_STOPPED;
          failMic(g);
        }
      }, 1500);
    });
  };
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
        card.clear();
        clearActivity();
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
        if (conn === "live") {
          micProblem = MIC_STOPPED;
          failMic(gen);
        }
      },
      onMediaStateChanged: (ms) => {
        if (gen !== generation || ms.mic.state !== "error") return;
        micProblem ??= micProblemNotice(ms.mic.reason, String(ms.mic.details ?? ""));
        if (conn === "live") {
          micProblem = MIC_STOPPED;
          failMic(gen);
        }
      },
      onTrackStarted: (track, participant) => {
        if (gen !== generation || track.kind !== "audio") return;
        if (participant?.local) {
          watchMic(gen, track);
        } else {
          attachBotAudio(track);
        }
      },
      onUserStartedSpeaking: () => setPhase(gen, "userSpeaking"),
      // Ignore out-of-order events: only leave the phase the event belongs to.
      onUserStoppedSpeaking: () => {
        if (phase === "userSpeaking") setPhase(gen, "thinking");
      },
      // Bot output (prose + code) arrives before audio starts, so the turn
      // boundary is the LLM response start, not "started speaking".
      onBotLlmStarted: () => transcript.startBotTurn(),
      onBotStartedSpeaking: () => {
        setPhase(gen, "speaking");
      },
      onBotStoppedSpeaking: () => {
        if (phase === "speaking") setPhase(gen, "listening");
        void resyncActions(gen);
      },
      onServerMessage: (data: unknown) => handleServerMessage(gen, data),
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
      // Each sentence arrives twice (spoken=false before TTS, spoken=true
      // after); render the early one so prose and code keep their order.
      onBotOutput: (d) => {
        if (gen === generation) transcript.botOutput(d);
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
  void resyncActions(gen);

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

// ---------- assistant actions (confirmation card + tool activity) ----------

const ACTIVITY_LABELS: Record<string, string> = {
  web_search: "Searching the web\u2026",
  fetch_page: "Reading a page\u2026",
  list_dir: "Looking at your files\u2026",
  find_file: "Looking at your files\u2026",
  read_file: "Looking at your files\u2026",
  file_info: "Looking at your files\u2026",
  move_file: "Preparing a change\u2026",
  edit_file: "Preparing a change\u2026",
};
const ACTIVITY_TIMEOUT_MS = 20000;
let activityTimer: number | undefined;

function clearActivity(): void {
  window.clearTimeout(activityTimer);
  activityEl.textContent = "";
}

function setActivity(name: string): void {
  activityEl.textContent = ACTIVITY_LABELS[name] ?? "Working\u2026";
  window.clearTimeout(activityTimer);
  activityTimer = window.setTimeout(clearActivity, ACTIVITY_TIMEOUT_MS);
}

function asAction(v: unknown): PublicAction | null {
  const a = v as Partial<PublicAction> | null;
  if (!a || typeof a !== "object") return null;
  if (typeof a.id !== "string" || !a.id) return null;
  if (a.kind !== "move" && a.kind !== "edit") return null;
  if (typeof a.summary !== "string") return null;
  if (a.diff !== null && a.diff !== undefined && typeof a.diff !== "string") return null;
  return {
    id: a.id,
    kind: a.kind,
    summary: a.summary,
    diff: a.diff ?? null,
    expires_in: typeof a.expires_in === "number" ? a.expires_in : 300,
  };
}

let resyncing = false;
/** A card can exist server-side without the UI knowing: ask, and show the first one. */
async function resyncActions(gen: number): Promise<void> {
  if (resyncing || gen !== generation || conn !== "live" || card.isOpen) return;
  resyncing = true;
  try {
    const list = await pendingActions();
    if (gen !== generation || conn !== "live" || card.isOpen) return;
    for (const item of list) {
      const a = asAction(item);
      if (a && card.show(a)) break;
    }
  } catch (e) {
    console.warn("could not fetch pending actions", e);
  } finally {
    resyncing = false;
  }
}

function resultFor(status: string, kind: PublicAction["kind"] | null, summary: unknown): string {
  switch (status) {
    case "done":
      return kind === "edit" ? RESULT_TEXT.edited : kind === "move" ? RESULT_TEXT.moved : "Done.";
    case "denied":
      return RESULT_TEXT.denied;
    case "expired":
      return RESULT_TEXT.expired;
    default:
      return typeof summary === "string" && summary ? summary.slice(0, 200) : RESULT_TEXT.failed;
  }
}

function handleServerMessage(gen: number, data: unknown): void {
  if (gen !== generation) return;
  const m = data as { type?: string; [k: string]: unknown } | null;
  if (!m || typeof m !== "object") return;
  switch (m.type) {
    case "pending_action": {
      const a = asAction(m.action);
      if (!a) {
        console.warn("malformed pending_action ignored");
        return;
      }
      card.show(a); // never replaces a showing card; logs and ignores
      return;
    }
    case "tool_activity":
      if (typeof m.name !== "string") return;
      if (m.state === "start") {
        setActivity(m.name);
      } else if (m.state === "end") {
        clearActivity();
        void resyncActions(gen);
      }
      return;
    case "action_result": {
      const status = String(m.status);
      if (!["done", "denied", "expired", "failed"].includes(status) || typeof m.id !== "string") return;
      if (card.currentId === m.id) {
        const text = resultFor(status, card.currentKind, m.summary);
        card.finish(true);
        card.showResult(text);
      } else {
        card.markClosed(m.id);
        card.showResult(resultFor(status, null, m.summary));
      }
      return;
    }
    case "actions_cleared":
      card.clear();
      return;
  }
}

card.onDecision(async (id, decision) => {
  try {
    await (decision === "approve" ? approveAction(id) : denyAction(id));
    if (card.currentId === id) {
      const kind = card.currentKind;
      card.finish(true);
      card.showResult(decision === "deny" ? RESULT_TEXT.denied : kind === "edit" ? RESULT_TEXT.edited : RESULT_TEXT.moved);
    }
  } catch (e) {
    if (card.currentId !== id) return;
    if (e instanceof ApiError && (e.status === 404 || e.status === 409)) {
      card.finish(true);
      card.showResult(RESULT_TEXT.stale);
    } else if (e instanceof ApiError && e.status === 422) {
      card.finish(true);
      card.showResult(e.detail);
    } else if (e instanceof ApiError) {
      card.reenable(e.detail || "Something went wrong. Try again.");
    } else {
      card.reenable("Network error \u2014 try again.");
    }
  }
  void resyncActions(generation);
});
card.onExpired(() => void resyncActions(generation));

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
  if (card.handleKey(e)) return; // Enter/Esc while a confirmation is open (Esc never interrupts)
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

// ---------- speaker test ----------

function beepWavUrl(): string {
  const rate = 24000;
  const n = Math.floor(rate * 0.4);
  const buf = new ArrayBuffer(44 + n * 2);
  const v = new DataView(buf);
  const str = (o: number, s: string) => [...s].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)));
  str(0, "RIFF");
  v.setUint32(4, 36 + n * 2, true);
  str(8, "WAVEfmt ");
  v.setUint32(16, 16, true);
  v.setUint16(20, 1, true);
  v.setUint16(22, 1, true);
  v.setUint32(24, rate, true);
  v.setUint32(28, rate * 2, true);
  v.setUint16(32, 2, true);
  v.setUint16(34, 16, true);
  str(36, "data");
  v.setUint32(40, n * 2, true);
  for (let i = 0; i < n; i++) {
    const fade = Math.min(1, i / 400, (n - i) / 400); // avoid clicks
    v.setInt16(44 + i * 2, Math.sin((2 * Math.PI * 440 * i) / rate) * 0.4 * fade * 32767, true);
  }
  return URL.createObjectURL(new Blob([buf], { type: "audio/wav" }));
}

const beepNote = $("beep-note");
$("beep-btn").addEventListener("click", (ev) => {
  (ev.currentTarget as HTMLButtonElement).blur();
  const url = beepWavUrl();
  const a = new Audio(url); // same media-element output path as the bot voice
  a.volume = 1;
  beepNote.hidden = false;
  a.addEventListener("ended", () => URL.revokeObjectURL(url));
  a.play().then(
    () => {
      beepNote.textContent =
        "If you heard the beep, your speaker works — if not, check the output device and volume.";
    },
    (e) => {
      URL.revokeObjectURL(url);
      console.warn("beep play() rejected", e);
      beepNote.textContent = `The browser blocked playback (${(e as Error).name}). Click Test speaker again or check site sound settings.`;
    },
  );
});

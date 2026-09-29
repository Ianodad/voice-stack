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
    /permission|not allowed|denied|getUserMedia/i.test(msg)
  );
}

// ---------- connection ----------

async function refreshSidebar(): Promise<void> {
  try {
    sidebar.setConversations(await listConversations());
  } catch (e) {
    console.warn("could not list conversations", e);
  }
}

async function teardown(): Promise<void> {
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
      onUserStartedSpeaking: () => setPhase(gen, "userSpeaking"),
      onUserStoppedSpeaking: () => setPhase(gen, "thinking"),
      onBotStartedSpeaking: () => setPhase(gen, "speaking"),
      onBotStoppedSpeaking: () => setPhase(gen, "listening"),
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
  conn = "live";
  phase = "listening";
  render();

  // X-Conversation-Id is not readable via client-js: pick the newest one.
  try {
    const list = await listConversations();
    if (gen !== generation) return;
    sidebar.setConversations(list);
    if (!id && list[0]) {
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
  const id = activeId;
  generation++;
  conn = "connecting";
  showNotice(null);
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
  muted = !muted;
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
});

void refreshSidebar().then(render);
render();

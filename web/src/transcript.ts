import type { StoredMessage } from "./api";

export type Part = { type: "text" | "code"; language?: string; text: string };

const FENCE = "```";

/**
 * Split text into prose and fenced-code parts. Shared by live bot-output
 * events and stored history. An unterminated fence is still a code part.
 */
export function parseFences(text: string): Part[] {
  const parts: Part[] = [];
  const pushText = (t: string) => {
    if (t.trim()) parts.push({ type: "text", text: t.trim() });
  };
  let rest = text;
  for (;;) {
    const open = rest.indexOf(FENCE);
    if (open < 0) {
      pushText(rest);
      return parts;
    }
    pushText(rest.slice(0, open));
    rest = rest.slice(open + FENCE.length);
    const nl = rest.indexOf("\n");
    const inlineClose = rest.indexOf(FENCE);
    let language = "";
    let body: string;
    if (nl < 0 || (inlineClose >= 0 && inlineClose < nl)) {
      // "```code```" on one line (or an unterminated opener with no newline).
      if (inlineClose >= 0) {
        parts.push({ type: "code", language, text: rest.slice(0, inlineClose) });
        rest = rest.slice(inlineClose + FENCE.length);
        continue;
      }
      language = rest.trim();
      parts.push({ type: "code", language, text: "" });
      return parts;
    }
    language = rest.slice(0, nl).trim();
    body = rest.slice(nl + 1);
    const close = body.indexOf(FENCE);
    if (close < 0) {
      parts.push({ type: "code", language, text: body.replace(/\n$/, "") });
      return parts;
    }
    parts.push({ type: "code", language, text: body.slice(0, close).replace(/\n$/, "") });
    rest = body.slice(close + FENCE.length);
  }
}

function copyText(text: string): Promise<void> {
  if (navigator.clipboard?.writeText) {
    return navigator.clipboard.writeText(text).catch(() => legacyCopy(text));
  }
  return legacyCopy(text);
}

function legacyCopy(text: string): Promise<void> {
  const ta = document.createElement("textarea");
  ta.value = text;
  ta.setAttribute("readonly", "");
  ta.style.position = "fixed";
  ta.style.opacity = "0";
  document.body.append(ta);
  ta.select();
  let ok = false;
  try {
    ok = document.execCommand("copy");
  } catch {
    ok = false;
  }
  ta.remove();
  return ok ? Promise.resolve() : Promise.reject(new Error("copy failed"));
}

/** A dark code box with a language label and a Copy button. textContent only. */
export function codeBlock(language: string, code: string): HTMLElement {
  const box = document.createElement("div");
  box.className = "codebox";
  const bar = document.createElement("div");
  bar.className = "codebar";
  const lang = document.createElement("span");
  lang.className = "codelang";
  lang.textContent = language || "code";
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "copybtn";
  btn.textContent = "Copy";
  btn.setAttribute("aria-label", "Copy code");
  let timer: number | undefined;
  btn.addEventListener("click", () => {
    void copyText(code).then(
      () => {
        btn.textContent = "Copied";
        window.clearTimeout(timer);
        timer = window.setTimeout(() => (btn.textContent = "Copy"), 1500);
      },
      () => {
        btn.textContent = "Copy failed";
        window.clearTimeout(timer);
        timer = window.setTimeout(() => (btn.textContent = "Copy"), 1500);
      },
    );
  });
  bar.append(lang, btn);
  const pre = document.createElement("pre");
  pre.className = "codepre";
  const el = document.createElement("code");
  el.textContent = code;
  pre.append(el);
  box.append(bar, pre);
  return box;
}

export class Transcript {
  private botBody: HTMLElement | null = null;
  private botText: HTMLElement | null = null; // trailing prose span of the current bot line

  constructor(private root: HTMLElement) {}

  clear(): void {
    this.root.replaceChildren();
    this.botBody = null;
    this.botText = null;
  }

  showHistory(messages: StoredMessage[]): void {
    this.clear();
    for (const m of messages) {
      const body = this.line(m.role === "user" ? "user" : "bot");
      if (m.role === "user") body.textContent = m.content;
      else this.addParts(body, parseFences(m.content));
    }
    this.botBody = null;
    this.botText = null;
  }

  /** A new bot turn begins: the next bot output starts a fresh line. */
  startBotTurn(): void {
    this.botBody = null;
    this.botText = null;
  }

  addUser(text: string): void {
    this.startBotTurn();
    this.line("user").textContent = text;
  }

  /** Handle one RTVI bot-output event: code renders as a box, prose as text.
   *  Each sentence arrives twice (spoken=false before TTS, spoken=true after);
   *  the early one is rendered so prose and code keep their order. */
  botOutput(d: { text: string; aggregated_by?: string; spoken?: boolean }): void {
    if (!d.text.trim()) return;
    if (d.aggregated_by === "code") this.appendBotCode(d.text);
    else if (d.spoken !== true) this.appendBot(d.text);
  }

  /** Append prose to the current bot line (starting one if needed). */
  appendBot(text: string): void {
    this.addParts(this.ensureBot(), [{ type: "text", text: text.trim() }], true);
  }

  /** A raw fenced code event (or any text containing fences) from the bot. */
  appendBotCode(raw: string): void {
    this.addParts(this.ensureBot(), parseFences(raw), true);
  }

  private ensureBot(): HTMLElement {
    if (!this.botBody) this.botBody = this.line("bot");
    return this.botBody;
  }

  private addParts(body: HTMLElement, parts: Part[], live = false): void {
    for (const p of parts) {
      if (p.type === "code") {
        body.append(codeBlock(p.language ?? "", p.text));
        if (live) this.botText = null;
      } else {
        let span = live ? this.botText : null;
        if (span && body.contains(span)) {
          span.textContent = `${span.textContent ?? ""} ${p.text}`;
        } else {
          span = document.createElement("span");
          span.className = "prose";
          span.textContent = p.text;
          body.append(span);
          if (live) this.botText = span;
        }
      }
    }
    this.scroll();
  }

  private line(kind: "user" | "bot"): HTMLElement {
    const el = document.createElement("div");
    el.className = `line ${kind}`;
    const who = document.createElement("span");
    who.className = "who";
    who.textContent = kind === "user" ? "You" : "Assistant";
    const body = document.createElement("div");
    body.className = "body";
    el.append(who, body);
    this.root.append(el);
    this.scroll();
    return body;
  }

  private scroll(): void {
    this.root.scrollTop = this.root.scrollHeight;
  }
}

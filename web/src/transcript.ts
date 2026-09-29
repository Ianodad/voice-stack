import type { StoredMessage } from "./api";

export class Transcript {
  private botLine: HTMLElement | null = null;

  constructor(private root: HTMLElement) {}

  clear(): void {
    this.root.replaceChildren();
    this.botLine = null;
  }

  showHistory(messages: StoredMessage[]): void {
    this.clear();
    for (const m of messages) this.line(m.role === "user" ? "user" : "bot", m.content);
    this.botLine = null;
  }

  /** A new bot turn begins: the next appendBot starts a fresh line. */
  startBotTurn(): void {
    this.botLine = null;
  }

  addUser(text: string): void {
    this.botLine = null;
    this.line("user", text);
  }

  /** Append streamed TTS text to the current bot line (starting one if needed). */
  appendBot(text: string): void {
    if (!this.botLine) {
      this.botLine = this.line("bot", "");
    }
    const cur = this.botLine.textContent ?? "";
    this.botLine.textContent = cur ? `${cur} ${text.trim()}` : text.trim();
    this.scroll();
  }

  private line(kind: "user" | "bot", text: string): HTMLElement {
    const el = document.createElement("p");
    el.className = `line ${kind}`;
    const who = document.createElement("span");
    who.className = "who";
    who.textContent = kind === "user" ? "You" : "Assistant";
    const body = document.createElement("span");
    body.className = "body";
    body.textContent = text;
    el.append(who, body);
    this.root.append(el);
    this.scroll();
    // botLine points at the element whose text we mutate: the body span.
    return body;
  }

  private scroll(): void {
    this.root.scrollTop = this.root.scrollHeight;
  }
}

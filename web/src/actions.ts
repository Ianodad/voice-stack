// Confirmation card for assistant actions (move / edit).
//
// Security-critical: the card is the user's only defence against file changes
// caused by prompt-injected text. Every server-provided string is rendered via
// textContent after escapeInvisible(), never innerHTML.

export type PublicAction = {
  id: string;
  kind: "move" | "edit";
  summary: string;
  diff: string | null;
  expires_in: number;
};

export type Decision = "approve" | "deny";
export type Segment = { kind: "text" | "esc"; text: string };

export const ARM_DELAY_MS = 500;
export const RESULT_MS = 4000;

// Characters that are invisible, reorder text, or look like a normal space.
const SUSPICIOUS =
  /[\p{Cc}\p{Cf}\p{Zl}\p{Zp}\p{Zs}\p{Cs}\p{Co}\p{Cn}\p{Variation_Selector}ㅤᅟᅠﾠ⠀]/u;

function escapeOf(cp: number): string {
  const hex = cp.toString(16).toUpperCase();
  return cp > 0xffff ? `\\u{${hex}}` : `\\u${hex.padStart(4, "0")}`;
}

/**
 * Split `s` into plain text and visible escape chips. Anything invisible or
 * confusable becomes `\uXXXX`; a real backslash becomes `\\` (as a chip) so a
 * typed-out `‮` can never be mistaken for a real escape.
 * Kept: \n, \t, and the ordinary space.
 */
export function escapeInvisible(s: string): Segment[] {
  const out: Segment[] = [];
  let buf = "";
  const flush = () => {
    if (buf) out.push({ kind: "text", text: buf });
    buf = "";
  };
  for (const ch of s) {
    // for..of yields a lone surrogate as a single code unit.
    const cp = ch.codePointAt(0)!;
    if (ch === "\n" || ch === "\t" || ch === " ") {
      buf += ch;
    } else if (ch === "\\") {
      flush();
      out.push({ kind: "esc", text: "\\\\" });
    } else if (SUSPICIOUS.test(ch)) {
      flush();
      out.push({ kind: "esc", text: escapeOf(cp) });
    } else {
      buf += ch;
    }
  }
  flush();
  return out;
}

/** Append `s` to `parent` as text nodes and `.esc` chips. textContent only. */
export function appendEscaped(parent: HTMLElement, s: string): void {
  for (const seg of escapeInvisible(s)) {
    if (seg.kind === "text") {
      parent.append(document.createTextNode(seg.text));
    } else {
      const chip = document.createElement("span");
      chip.className = "esc";
      chip.textContent = seg.text;
      parent.append(chip);
    }
  }
}

const RESULT_TEXT = {
  moved: "Moved.",
  edited: "Edited — backup saved.",
  denied: "Denied.",
  expired: "That request expired.",
  stale: "That request expired or was already handled.",
  failed: "That action failed.",
} as const;
export { RESULT_TEXT };

export class ActionCard {
  private readonly host: HTMLElement;
  private readonly resultEl: HTMLElement;
  private card: HTMLElement | null = null;
  private approveBtn: HTMLButtonElement | null = null;
  private denyBtn: HTMLButtonElement | null = null;
  private errorEl: HTMLElement | null = null;
  private current: PublicAction | null = null;
  private shownAt = 0;
  private armed = false;
  private decided = false;
  private armTimer: number | undefined;
  private expiryTimer: number | undefined;
  private resultTimer: number | undefined;
  private cb: ((id: string, d: Decision) => void) | null = null;
  private onExpire: (() => void) | null = null;
  /** Ids that reached a terminal state; a late resync must not resurrect them. */
  private readonly closed = new Set<string>();

  constructor(host: HTMLElement, resultEl: HTMLElement) {
    this.host = host;
    this.resultEl = resultEl;
  }

  onDecision(cb: (id: string, d: Decision) => void): void {
    this.cb = cb;
  }

  /** Called when the local 5-minute timer runs out (card already cleared). */
  onExpired(cb: () => void): void {
    this.onExpire = cb;
  }

  get isOpen(): boolean {
    return this.current !== null;
  }
  get currentId(): string | null {
    return this.current?.id ?? null;
  }
  get currentKind(): PublicAction["kind"] | null {
    return this.current?.kind ?? null;
  }
  contains(t: EventTarget | null): boolean {
    return !!this.card && t instanceof Node && this.card.contains(t);
  }

  show(a: PublicAction): boolean {
    if (this.current) {
      console.warn("pending_action ignored: a card is already showing", a.id);
      return false;
    }
    if (this.closed.has(a.id)) return false;
    this.current = a;
    this.decided = false;
    this.armed = false;
    this.shownAt = performance.now();

    const card = document.createElement("div");
    card.className = "action-card";
    card.setAttribute("role", "alertdialog");
    card.setAttribute("aria-live", "polite");
    card.tabIndex = -1;

    const title = document.createElement("h2");
    title.id = "action-card-title";
    title.textContent = a.kind === "edit" ? "Confirm edit" : "Confirm move";
    card.setAttribute("aria-labelledby", title.id);

    const summary = document.createElement("p");
    summary.className = "action-summary";
    appendEscaped(summary, a.summary);

    card.append(title, summary);

    if (a.kind === "edit" && a.diff) {
      const pre = document.createElement("pre");
      pre.className = "action-diff";
      const lines = a.diff.split("\n");
      lines.forEach((line, i) => {
        const span = document.createElement("span");
        const c = line.charAt(0);
        span.className = c === "+" ? "add" : c === "-" ? "del" : "ctx";
        appendEscaped(span, line);
        pre.append(span);
        if (i < lines.length - 1) pre.append(document.createTextNode("\n"));
      });
      card.append(pre);
    }

    this.errorEl = document.createElement("p");
    this.errorEl.className = "action-error";
    this.errorEl.hidden = true;

    const row = document.createElement("div");
    row.className = "action-buttons";
    this.approveBtn = this.mkButton("Approve", "Enter", "approve");
    this.approveBtn.classList.add("approve");
    this.denyBtn = this.mkButton("Deny", "Esc", "deny");
    const caption = document.createElement("span");
    caption.className = "action-caption muted small-text";
    caption.textContent = "expires in 5 minutes";
    row.append(this.approveBtn, this.denyBtn, caption);

    card.append(this.errorEl, row);
    this.card = card;
    this.setArming(true);
    this.host.replaceChildren(card);
    this.host.hidden = false;
    card.focus({ preventScroll: true });

    window.clearTimeout(this.armTimer);
    this.armTimer = window.setTimeout(() => {
      this.armed = true;
      if (!this.decided) this.setArming(false);
    }, ARM_DELAY_MS);

    window.clearTimeout(this.expiryTimer);
    if (Number.isFinite(a.expires_in) && a.expires_in > 0) {
      const id = a.id;
      this.expiryTimer = window.setTimeout(() => {
        if (this.current?.id !== id) return;
        this.finish(true);
        this.showResult(RESULT_TEXT.expired);
        this.onExpire?.();
      }, (a.expires_in + 2) * 1000);
    }
    return true;
  }

  private mkButton(label: string, key: string, d: Decision): HTMLButtonElement {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "btn";
    b.dataset.label = `${label} (${key})`;
    b.addEventListener("click", () => this.decide(d));
    return b;
  }

  private setArming(arming: boolean): void {
    for (const b of [this.approveBtn, this.denyBtn]) {
      if (!b) continue;
      b.disabled = arming;
      b.textContent = arming ? "arming…" : b.dataset.label ?? "";
    }
  }

  /** Returns true if a decision was started (used by the keyboard path). */
  decide(d: Decision): boolean {
    const a = this.current;
    if (!a || this.decided) return false;
    if (!this.armed || performance.now() - this.shownAt < ARM_DELAY_MS) return false;
    this.decided = true;
    if (this.approveBtn) this.approveBtn.disabled = true;
    if (this.denyBtn) this.denyBtn.disabled = true;
    if (this.errorEl) this.errorEl.hidden = true;
    this.cb?.(a.id, d);
    return true;
  }

  /** The POST failed before the server answered: keep the card, let the user retry. */
  reenable(message: string): void {
    if (!this.current) return;
    this.decided = false;
    this.setArming(false);
    if (this.errorEl) {
      this.errorEl.textContent = "";
      appendEscaped(this.errorEl, message);
      this.errorEl.hidden = false;
    }
  }

  /**
   * Keyboard entry point. Returns true when the event was consumed and must
   * not reach any other handler (in particular the Esc-interrupt one).
   */
  handleKey(e: KeyboardEvent): boolean {
    if (!this.current) return false;
    if (e.key !== "Enter" && e.key !== "Escape") return false;
    const t = e.target;
    const textField =
      t instanceof HTMLInputElement ||
      t instanceof HTMLTextAreaElement ||
      t instanceof HTMLSelectElement ||
      (t instanceof HTMLElement && t.isContentEditable);
    if (textField) return false;
    const foreignButton = t instanceof HTMLElement && !!t.closest("button, a[href]") && !this.contains(t);
    if (e.key === "Escape") {
      e.preventDefault();
      // Esc never interrupts while a card is open; it only denies when armed
      // and the event did not come from some other control.
      if (!e.repeat && !foreignButton) this.decide("deny");
      return true;
    }
    // Enter on one of the card's own buttons is a native click on that button:
    // don't also approve (Enter on Deny must mean Deny).
    if (this.contains(t) && t instanceof HTMLElement && t.closest("button")) return false;
    if (foreignButton) return false; // native click on the other control
    e.preventDefault();
    if (!e.repeat) this.decide("approve");
    return true;
  }

  /** Remove the card without marking the id terminal (teardown, actions_cleared). */
  clear(): void {
    this.finish(false);
  }

  /** Terminal: the id is finished and must not be shown again. */
  finish(terminal: boolean): void {
    if (this.current && terminal) this.closed.add(this.current.id);
    this.dropCard();
  }

  markClosed(id: string): void {
    this.closed.add(id);
  }

  private dropCard(): void {
    window.clearTimeout(this.armTimer);
    window.clearTimeout(this.expiryTimer);
    const hadFocus = !!this.card && this.card.contains(document.activeElement);
    this.card = null;
    this.approveBtn = this.denyBtn = null;
    this.errorEl = null;
    this.current = null;
    this.decided = false;
    this.armed = false;
    this.host.replaceChildren();
    this.host.hidden = true;
    if (hadFocus && document.activeElement instanceof HTMLElement) document.activeElement.blur();
  }

  /** Result text in the aria-live region for ~4 s. */
  showResult(text: string): void {
    window.clearTimeout(this.resultTimer);
    this.resultEl.textContent = "";
    appendEscaped(this.resultEl, text);
    this.resultTimer = window.setTimeout(() => {
      this.resultEl.textContent = "";
    }, RESULT_MS);
  }
}

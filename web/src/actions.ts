// Confirmation card for assistant actions (move / edit).
//
// Security-critical: the card is the user's only defence against file changes
// caused by prompt-injected text. Every server-provided string is rendered via
// textContent after parseServerText(), never innerHTML, and the card must never
// display something other than what will happen.

export type PublicAction = {
  id: string;
  kind: "move" | "edit";
  summary: string;
  diff: string | null;
  expires_in: number;
};

export type Decision = "approve" | "deny";
export type Segment = { kind: "text" | "esc"; text: string; spaces?: boolean; label?: string };

export const ARM_DELAY_MS = 500;
export const RESULT_MS = 4000;
/** Leading indentation up to this many spaces is shown literally; longer runs get a chip. */
export const MAX_LEADING_SPACES = 16;

// Characters that are invisible, reorder text, or look like a normal space.
// (Marks are handled separately: they are only dangerous in runs / after spaces.)
const SUSPICIOUS =
  /[\p{Cc}\p{Cf}\p{Zl}\p{Zp}\p{Zs}\p{Cs}\p{Co}\p{Cn}\p{Variation_Selector}\p{Default_Ignorable_Code_Point}ㅤᅟᅠﾠ⠀]/u;
const MARK = /[\p{Mn}\p{Me}]/u;

function escapeOf(cp: number): string {
  const hex = cp.toString(16);
  return cp > 0xffff ? `\\U${hex.padStart(8, "0")}` : `\\u${hex.padStart(4, "0")}`;
}

type Token = { esc: boolean; s: string };

/** Screen-reader text for an escape chip such as `\u202e`, `\U000e0020` or `\t`. */
function labelOf(esc: string): string {
  if (esc === "\\t") return "tab character";
  const cp = parseInt(esc.slice(2), 16);
  return `hidden character U+${cp.toString(16).toUpperCase().padStart(4, "0")}`;
}

/**
 * Server display contract: a real backslash arrives as `\\`; hidden chars as
 * `\uxxxx` / `\Uxxxxxxxx`; TAB as `\t`. Split into literal text and escape
 * tokens. Malformed escapes (lone `\`, `\u12`) stay literal. Backslashes are
 * NEVER re-escaped here.
 */
function tokenize(s: string): Token[] {
  const out: Token[] = [];
  let buf = "";
  const flush = () => {
    if (buf) out.push({ esc: false, s: buf });
    buf = "";
  };
  let i = 0;
  while (i < s.length) {
    const c = s[i];
    if (c !== "\\") {
      buf += c;
      i++;
      continue;
    }
    const n = s[i + 1];
    if (n === "\\") {
      buf += "\\";
      i += 2;
    } else if (n === "t") {
      flush();
      out.push({ esc: true, s: "\\t" });
      i += 2;
    } else if (n === "u" && /^[0-9a-fA-F]{4}$/.test(s.slice(i + 2, i + 6))) {
      flush();
      out.push({ esc: true, s: s.slice(i, i + 6) });
      i += 6;
    } else if (n === "U" && /^[0-9a-fA-F]{8}$/.test(s.slice(i + 2, i + 10)) && parseInt(s.slice(i + 2, i + 10), 16) <= 0x10ffff) {
      flush();
      out.push({ esc: true, s: s.slice(i, i + 10) });
      i += 10;
    } else {
      buf += "\\";
      i++;
    }
  }
  flush();
  return out;
}

/**
 * Parse a server display string into text and escape-chip segments. Also
 * escapes anything invisible the server missed (defence in depth), chips runs
 * of 3+ spaces (so a payload cannot hide behind padding), and chips marks that
 * are stacked (>2) or follow whitespace.
 */
export function parseServerText(s: string, maxLeadingSpaces = MAX_LEADING_SPACES): Segment[] {
  const out: Segment[] = [];
  let buf = "";
  let prev = " ";
  let run = 0;
  let spaces = 0;
  let seen = false; // any non-space content so far (leading-indent detection)
  const flush = () => {
    if (buf) out.push({ kind: "text", text: buf });
    buf = "";
  };
  const flushSpaces = (atEnd: boolean) => {
    if (!spaces) return;
    const leadingOk = !seen && spaces <= maxLeadingSpaces && !atEnd;
    if (spaces >= 3 && !leadingOk) {
      flush();
      out.push({ kind: "esc", text: `sp\u00d7${spaces}`, spaces: true, label: `${spaces} spaces` });
    } else {
      buf += " ".repeat(spaces);
    }
    spaces = 0;
    prev = " ";
  };
  for (const tok of tokenize(s)) {
    if (tok.esc) {
      flushSpaces(false);
      flush();
      out.push({ kind: "esc", text: tok.s, label: labelOf(tok.s) });
      seen = true;
      prev = "x";
      run = 0;
      continue;
    }
    for (const ch of tok.s) {
      if (ch === " ") {
        spaces++;
        run = 0;
        continue;
      }
      flushSpaces(false);
      const mark = MARK.test(ch);
      run = mark ? run + 1 : 0;
      const bad =
        SUSPICIOUS.test(ch) || (mark && (/\s/u.test(prev) || prev === "/" || run > 2));
      if (bad) {
        flush();
        const e = escapeOf(ch.codePointAt(0)!);
        out.push({ kind: "esc", text: e, label: labelOf(e) });
      } else {
        buf += ch;
      }
      seen = true;
      prev = ch;
    }
  }
  flushSpaces(true);
  flush();
  return out;
}

function appendSegments(parent: HTMLElement, segs: Segment[]): void {
  for (const seg of segs) {
    if (seg.kind === "text") {
      parent.append(document.createTextNode(seg.text));
    } else {
      const chip = document.createElement("span");
      chip.className = seg.spaces ? "esc spaces" : "esc";
      chip.textContent = seg.text;
      if (seg.label) {
        chip.setAttribute("role", "img");
        chip.setAttribute("aria-label", seg.label);
      }
      parent.append(chip);
    }
  }
}

/** Append a server display string to `parent`: text nodes and `.esc` chips only. */
export function appendServerText(parent: HTMLElement, s: string, maxLeadingSpaces = MAX_LEADING_SPACES): void {
  appendSegments(parent, parseServerText(s, maxLeadingSpaces));
}

/** "N changed lines in M hunks" from a unified diff. */
export function diffStats(diff: string): { changed: number; hunks: number } {
  let changed = 0;
  let hunks = 0;
  let inHunk = false;
  for (const line of diff.split("\n")) {
    if (line.startsWith("@@")) {
      hunks++;
      inHunk = true;
    } else if (inHunk && (line[0] === "+" || line[0] === "-")) {
      changed++;
    }
  }
  return { changed, hunks };
}

export function diffHeading(diff: string): string {
  const { changed, hunks } = diffStats(diff);
  const l = changed === 1 ? "1 changed line" : `${changed} changed lines`;
  const h = hunks === 1 ? "1 hunk" : `${hunks} hunks`;
  return `${l} in ${h}`;
}

function expiryCaption(sec: number): string {
  if (!Number.isFinite(sec) || sec <= 0) return "expires soon";
  if (sec < 60) return "expires in less than a minute";
  const m = Math.ceil(sec / 60);
  return `expires in ${m} minute${m === 1 ? "" : "s"}`;
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
  private captionEl: HTMLElement | null = null;
  private diffEl: HTMLElement | null = null;
  private hintEl: HTMLElement | null = null;
  private current: PublicAction | null = null;
  private expiresText = "";
  // Approve gating: all of these must hold.
  private timeArmed = false;
  private armStart = 0;
  /** Per scroll container: last geometry and whether its end has been seen. */
  private readonly seen = new Map<HTMLElement, { geom: string; ok: boolean }>();
  private lockedBy: "diff" | "card" | null = null;
  private decided = false;
  private armTimer: number | undefined;
  private expiryTimer: number | undefined;
  private resultTimer: number | undefined;
  private cb: ((id: string, d: Decision) => void) | null = null;
  private onExpire: (() => void) | null = null;
  private readonly env = () => this.reevaluate();
  /** Ids that reached a terminal state; a late resync must not resurrect them. */
  private readonly closed = new Set<string>();

  constructor(host: HTMLElement, resultEl: HTMLElement) {
    this.host = host;
    this.resultEl = resultEl;
  }

  onDecision(cb: (id: string, d: Decision) => void): void {
    this.cb = cb;
  }

  /** Called when the local expiry timer runs out (card already cleared). */
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
  isClosed(id: string): boolean {
    return this.closed.has(id);
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
    this.timeArmed = false;
    this.armStart = 0;
    this.seen.clear();
    this.lockedBy = null;
    this.expiresText = expiryCaption(a.expires_in);

    const card = document.createElement("div");
    card.className = "action-card";
    card.setAttribute("role", "alertdialog");
    card.tabIndex = -1;

    const title = document.createElement("h2");
    title.id = "action-card-title";
    title.textContent = a.kind === "edit" ? "Confirm edit" : "Confirm move";
    card.setAttribute("aria-labelledby", title.id);

    const summary = document.createElement("p");
    summary.className = "action-summary srv";
    summary.id = "action-card-summary";
    appendServerText(summary, a.summary);
    card.setAttribute("aria-describedby", summary.id);

    card.append(title, summary);

    if (a.kind === "edit" && a.diff) {
      const stats = document.createElement("p");
      stats.className = "action-stats";
      stats.textContent = diffHeading(a.diff);
      const pre = document.createElement("pre");
      pre.className = "action-diff srv";
      pre.tabIndex = 0;
      pre.setAttribute("role", "region");
      pre.setAttribute("aria-label", "Changes");
      for (const line of a.diff.split("\n")) {
        const span = document.createElement("span");
        const c = line.charAt(0);
        span.className = c === "+" ? "add" : c === "-" ? "del" : "ctx";
        if (c === "+" || c === "-" || c === " ") {
          // The marker is plain; the rest is server text (leading indent measured after it).
          span.append(document.createTextNode(c));
          appendServerText(span, line.slice(1));
        } else {
          appendServerText(span, line);
        }
        pre.append(span);
      }
      pre.addEventListener("scroll", this.env, { passive: true });
      this.diffEl = pre;
      card.append(stats, pre);
    }
    card.addEventListener("scroll", this.env, { passive: true });
    this.hintEl = document.createElement("p");
    this.hintEl.className = "action-hint";
    this.hintEl.hidden = true;
    card.append(this.hintEl);

    this.errorEl = document.createElement("p");
    this.errorEl.className = "action-error srv";
    this.errorEl.hidden = true;

    const row = document.createElement("div");
    row.className = "action-buttons";
    this.approveBtn = this.mkButton("Approve", "Enter", "approve");
    this.approveBtn.classList.add("approve");
    this.denyBtn = this.mkButton("Deny", "Esc", "deny");
    this.captionEl = document.createElement("span");
    this.captionEl.className = "action-caption muted small-text";
    row.append(this.approveBtn, this.denyBtn, this.captionEl);

    card.append(this.errorEl, row);
    this.card = card;
    this.host.replaceChildren(card);
    this.host.hidden = false;
    card.scrollIntoView({ block: "nearest" });
    card.focus({ preventScroll: true }); // the container, never Approve

    document.addEventListener("visibilitychange", this.env);
    window.addEventListener("focus", this.env);
    window.addEventListener("blur", this.env);
    window.addEventListener("resize", this.env);
    this.reevaluate();

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
    b.textContent = b.dataset.label;
    b.addEventListener("click", () => this.decide(d));
    return b;
  }

  /** Approve only counts while the user could actually see and act on the card. */
  private envOk(): boolean {
    if (document.visibilityState !== "visible" || !document.hasFocus()) return false;
    const b = this.approveBtn;
    if (!b) return false;
    const r = b.getBoundingClientRect();
    if (!(r.width > 0 && r.top >= 0 && r.left >= 0 && r.bottom <= window.innerHeight && r.right <= window.innerWidth)) return false;
    // Also inside the card's own (possibly scrolled) visible area.
    const c = this.card?.getBoundingClientRect();
    return !!c && r.top >= c.top && r.bottom <= c.bottom;
  }

  /**
   * "Everything the user must see has been seen": every scroll container (the
   * card and the diff) either fits or has been scrolled to its end. A latch per
   * container, reset whenever its geometry changes (resize / zoom) unless it is
   * at the end or fits.
   */
  private updateSeen(): boolean {
    this.lockedBy = null;
    const els: Array<["diff" | "card", HTMLElement | null]> = [["diff", this.diffEl], ["card", this.card]];
    for (const [name, el] of els) {
      if (!el) continue;
      const needs = el.scrollHeight - el.clientHeight > 2;
      const end = el.scrollTop + el.clientHeight >= el.scrollHeight - 2;
      const geom = `${el.scrollHeight}:${el.clientHeight}`;
      const st = this.seen.get(el) ?? { geom: "", ok: false };
      if (geom !== st.geom) {
        st.geom = geom;
        st.ok = !needs || end;
      } else if (!needs || end) {
        st.ok = true;
      }
      this.seen.set(el, st);
      if (!st.ok && this.lockedBy === null) this.lockedBy = name;
    }
    return this.lockedBy === null;
  }

  private canApprove(): boolean {
    return (
      !this.decided &&
      this.timeArmed &&
      performance.now() - this.armStart >= ARM_DELAY_MS &&
      this.updateSeen() &&
      this.envOk()
    );
  }

  /** Re-check visibility/focus/scroll; (re)start the arm delay when conditions break. */
  private reevaluate(): void {
    if (!this.current || this.decided) return;
    this.updateSeen();
    if (!this.envOk()) {
      window.clearTimeout(this.armTimer);
      this.armTimer = undefined;
      this.timeArmed = false;
    } else if (!this.timeArmed && this.armTimer === undefined) {
      this.armStart = performance.now();
      this.armTimer = window.setTimeout(() => {
        this.armTimer = undefined;
        if (this.envOk()) this.timeArmed = true;
        this.render();
      }, ARM_DELAY_MS);
    }
    this.render();
  }

  private render(): void {
    const a = this.approveBtn;
    const d = this.denyBtn;
    if (!a || !d) return;
    const can = this.canApprove();
    a.disabled = this.decided || !can;
    a.textContent = this.timeArmed ? a.dataset.label ?? "" : "arming…";
    d.disabled = this.decided; // Deny / Esc work immediately
    if (this.hintEl) {
      this.hintEl.hidden = this.lockedBy === null;
      this.hintEl.textContent =
        this.lockedBy === "diff" ? "Scroll to see the rest of the changes." : "Scroll down to see the whole request.";
    }
    if (this.captionEl) {
      this.captionEl.textContent = this.lockedBy !== null ? "scroll to the end to enable Approve" : this.expiresText;
    }
  }

  /** Returns true if a decision was started (used by the keyboard path). */
  decide(d: Decision): boolean {
    const a = this.current;
    if (!a || this.decided) return false;
    if (d === "approve" && !this.canApprove()) return false;
    this.decided = true;
    if (this.errorEl) this.errorEl.hidden = true;
    this.render();
    this.cb?.(a.id, d);
    return true;
  }

  /** The POST failed before the server answered: keep the card, let the user retry. */
  reenable(message: string): void {
    if (!this.current) return;
    this.decided = false;
    if (this.errorEl) {
      this.errorEl.textContent = "";
      appendServerText(this.errorEl, message);
      this.errorEl.hidden = false;
    }
    this.reevaluate();
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
      // Esc never interrupts while a card is open. It denies immediately
      // (not arm-delayed) unless it is a repeat, an IME key, or came from
      // some other control.
      if (!e.repeat && !e.isComposing && !foreignButton) this.decide("deny");
      return true;
    }
    if (e.isComposing || e.keyCode === 229) return false;
    if (e.metaKey || e.ctrlKey || e.altKey || e.shiftKey) return false; // never approve via a chord
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
    this.armTimer = undefined;
    document.removeEventListener("visibilitychange", this.env);
    window.removeEventListener("focus", this.env);
    window.removeEventListener("blur", this.env);
    window.removeEventListener("resize", this.env);
    const hadFocus = !!this.card && this.card.contains(document.activeElement);
    this.card = null;
    this.approveBtn = this.denyBtn = null;
    this.errorEl = this.captionEl = this.diffEl = this.hintEl = null;
    this.current = null;
    this.decided = false;
    this.timeArmed = false;
    this.host.replaceChildren();
    this.host.hidden = true;
    if (hadFocus && document.activeElement instanceof HTMLElement) document.activeElement.blur();
  }

  /** Result text in the single aria-live region for ~4 s. */
  showResult(text: string): void {
    window.clearTimeout(this.resultTimer);
    this.resultEl.textContent = "";
    appendServerText(this.resultEl, text);
    this.resultTimer = window.setTimeout(() => {
      this.resultEl.textContent = "";
    }, RESULT_MS);
  }
}

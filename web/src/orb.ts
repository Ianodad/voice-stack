export type OrbState =
  | "connecting"
  | "listening"
  | "userSpeaking"
  | "thinking"
  | "speaking"
  | "muted"
  | "error";

interface Palette {
  accent: string;
  user: string;
  bot: string;
  muted: string;
  error: string;
}

function readPalette(): Palette {
  const cs = getComputedStyle(document.documentElement);
  const v = (n: string) => cs.getPropertyValue(n).trim();
  return {
    accent: v("--orb-listen"),
    user: v("--orb-user"),
    bot: v("--orb-bot"),
    muted: v("--orb-muted"),
    error: v("--orb-error"),
  };
}

export class Orb {
  private ctx: CanvasRenderingContext2D;
  private state: OrbState = "connecting";
  private target = 0;
  private level = 0;
  private palette = readPalette();
  private raf = 0;
  private size: number;
  private reduced = matchMedia("(prefers-reduced-motion: reduce)");

  constructor(canvas: HTMLCanvasElement) {
    const dpr = window.devicePixelRatio || 1;
    this.size = canvas.width;
    canvas.style.width = `${this.size}px`;
    canvas.style.height = `${this.size}px`;
    canvas.width = this.size * dpr;
    canvas.height = this.size * dpr;
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("Canvas 2D unavailable");
    ctx.scale(dpr, dpr);
    this.ctx = ctx;
    matchMedia("(prefers-color-scheme: light)").addEventListener("change", () => {
      this.palette = readPalette();
    });
    this.raf = requestAnimationFrame(this.frame);
  }

  setState(s: OrbState): void {
    this.state = s;
  }

  getState(): OrbState {
    return this.state;
  }

  /** 0..1 audio level; smoothed inside the render loop. */
  setLevel(l: number): void {
    this.target = Number.isFinite(l) ? Math.max(0, Math.min(1, l)) : 0;
  }

  destroy(): void {
    cancelAnimationFrame(this.raf);
  }

  private frame = (t: number): void => {
    this.raf = requestAnimationFrame(this.frame);
    this.draw(t / 1000);
  };

  private draw(t: number): void {
    const { ctx, size, palette: p } = this;
    const s = this.state;
    const motion = this.reduced.matches ? 0.3 : 1;
    // Smooth the level: fast attack, slower release.
    const k = this.target > this.level ? 0.35 : 0.1;
    this.level += (this.target - this.level) * k;
    const lv = this.level;

    let color = p.accent;
    let radius = 0.28;
    let alpha = 1;
    let swirl = 0;
    switch (s) {
      case "connecting":
        color = p.accent;
        radius = 0.26 + 0.02 * Math.sin(t * 1.4) * motion;
        alpha = 0.35 + 0.15 * Math.sin(t * 1.4) * motion;
        break;
      case "listening":
        color = p.accent;
        radius = 0.28 + 0.015 * Math.sin(t * 1.8) * motion + 0.1 * lv;
        alpha = 0.85;
        break;
      case "userSpeaking":
        color = p.user;
        radius = 0.3 + 0.16 * lv;
        alpha = 1;
        break;
      case "thinking":
        color = p.accent;
        radius = 0.28 + 0.01 * Math.sin(t * 2) * motion;
        alpha = 0.9;
        swirl = 1;
        break;
      case "speaking":
        color = p.bot;
        radius = 0.3 + 0.15 * lv;
        alpha = 1;
        break;
      case "muted":
        color = p.muted;
        radius = 0.26;
        alpha = 0.6;
        break;
      case "error":
        color = p.error;
        radius = 0.26;
        alpha = 0.9;
        break;
    }

    ctx.clearRect(0, 0, size, size);
    const cx = size / 2;
    const cy = size / 2;
    const r = radius * size;

    // soft glow
    const gr = Math.min(r * 1.9, size / 2);
    const glow = ctx.createRadialGradient(cx, cy, r * 0.6, cx, cy, gr);
    glow.addColorStop(0, color);
    glow.addColorStop(1, "transparent");
    ctx.globalAlpha = 0.28 * alpha;
    ctx.fillStyle = glow;
    ctx.beginPath();
    ctx.arc(cx, cy, gr, 0, Math.PI * 2);
    ctx.fill();

    // core
    ctx.globalAlpha = alpha;
    const core = ctx.createRadialGradient(cx - r * 0.3, cy - r * 0.3, r * 0.1, cx, cy, r);
    core.addColorStop(0, color);
    core.addColorStop(1, color);
    ctx.fillStyle = core;
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fill();

    if (swirl) {
      // rotating arcs around the core
      ctx.globalAlpha = 0.9;
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.lineCap = "round";
      const a = t * 2.2 * motion;
      for (let i = 0; i < 3; i++) {
        const start = a + (i * Math.PI * 2) / 3;
        ctx.beginPath();
        ctx.arc(cx, cy, r * 1.35, start, start + 0.9);
        ctx.stroke();
      }
    }

    if (s === "error") {
      ctx.globalAlpha = 1;
      ctx.strokeStyle = p.error;
      ctx.lineWidth = 4;
      ctx.beginPath();
      ctx.arc(cx, cy, r * 1.45, 0, Math.PI * 2);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }
}

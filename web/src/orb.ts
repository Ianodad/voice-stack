export type OrbState =
  | "connecting"
  | "listening"
  | "userSpeaking"
  | "thinking"
  | "speaking"
  | "muted"
  | "error";

/**
 * A glowing multi-colour star: many thin rays radiate from a bright core.
 * Every ray has its own hue (spread over the colour wheel, slowly drifting).
 * All look parameters are eased toward per-state targets, so state changes
 * never snap. Canvas 2D only; no per-frame allocations beyond colour strings.
 */

const RAYS = 84;
const TAU = Math.PI * 2;

interface Params {
  active: number; // fraction of rays visible (0..1)
  bright: number; // overall alpha gain
  sat: number; // 0..100
  light: number; // 0..100 (dark theme; light theme is darkened at draw time)
  len: number; // base ray length as fraction of size
  audio: number; // how strongly the level stretches rays
  noise: number; // amount of per-ray length wobble
  spin: number; // radians / second
  wave: number; // travelling brightness wave strength
  err: number; // 0..1 shift to red
  flicker: number; // error flicker depth
  breath: number; // breathing depth
}

const TARGETS: Record<OrbState, Params> = {
  connecting: { active: 0.22, bright: 0.4, sat: 80, light: 62, len: 0.2, audio: 0, noise: 0.05, spin: 0, wave: 0, err: 0, flicker: 0, breath: 0.12 },
  listening: { active: 1, bright: 0.72, sat: 85, light: 62, len: 0.29, audio: 0.5, noise: 0.22, spin: 0.02, wave: 0, err: 0, flicker: 0, breath: 0.05 },
  userSpeaking: { active: 1, bright: 1, sat: 100, light: 60, len: 0.31, audio: 1.1, noise: 0.15, spin: 0.05, wave: 0, err: 0, flicker: 0, breath: 0 },
  thinking: { active: 1, bright: 0.65, sat: 90, light: 62, len: 0.3, audio: 0, noise: 0.1, spin: 0.35, wave: 1, err: 0, flicker: 0, breath: 0.03 },
  speaking: { active: 1, bright: 1, sat: 100, light: 62, len: 0.31, audio: 1.2, noise: 0.15, spin: 0.06, wave: 0, err: 0, flicker: 0, breath: 0 },
  muted: { active: 1, bright: 0.28, sat: 0, light: 55, len: 0.22, audio: 0, noise: 0.06, spin: 0, wave: 0, err: 0, flicker: 0, breath: 0.03 },
  error: { active: 1, bright: 0.75, sat: 95, light: 55, len: 0.26, audio: 0, noise: 0.08, spin: 0, wave: 0, err: 1, flicker: 0.5, breath: 0 },
};

function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}

export class Orb {
  private ctx: CanvasRenderingContext2D;
  private state: OrbState = "connecting";
  private target = 0;
  private level = 0;
  private raf = 0;
  private destroyed = false;
  private size: number;
  private dpr: number;
  private reduced = matchMedia("(prefers-reduced-motion: reduce)");
  private lightMq = matchMedia("(prefers-color-scheme: light)");
  private light = this.lightMq.matches;
  private cur: Params = { ...TARGETS.connecting };
  private lastT = -1;
  private rot = 0;
  private tm = 0; // motion-scaled clock

  // per-ray constants
  private ang = new Float32Array(RAYS);
  private hue = new Float32Array(RAYS);
  private ph1 = new Float32Array(RAYS);
  private ph2 = new Float32Array(RAYS);
  private w1 = new Float32Array(RAYS);
  private w2 = new Float32Array(RAYS);
  private base = new Float32Array(RAYS);
  private thr = new Float32Array(RAYS);

  private coreWhite: HTMLCanvasElement;
  private coreRed: HTMLCanvasElement;
  private coreDeep: HTMLCanvasElement;

  private onScheme = (): void => {
    this.light = this.lightMq.matches;
  };

  constructor(canvas: HTMLCanvasElement) {
    const dpr = window.devicePixelRatio || 1;
    this.dpr = dpr;
    this.size = canvas.width;
    canvas.style.width = `${this.size}px`;
    canvas.style.height = `${this.size}px`;
    canvas.width = Math.round(this.size * dpr);
    canvas.height = Math.round(this.size * dpr);
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("Canvas 2D unavailable");
    this.ctx = ctx;

    // deterministic pseudo-random so the star looks the same every load
    let seed = 1234567;
    const rnd = (): number => {
      seed = (seed * 1664525 + 1013904223) >>> 0;
      return seed / 4294967296;
    };
    for (let i = 0; i < RAYS; i++) {
      this.ang[i] = (i / RAYS) * TAU + (rnd() - 0.5) * 0.05;
      // hues interleaved so neighbours differ a bit, while still sweeping the wheel
      this.hue[i] = ((i / RAYS) * 360 + (rnd() - 0.5) * 40 + 360) % 360;
      this.ph1[i] = rnd() * TAU;
      this.ph2[i] = rnd() * TAU;
      this.w1[i] = 0.5 + rnd() * 0.9;
      this.w2[i] = 1.3 + rnd() * 1.6;
      // long/short mix gives the spiky star silhouette
      const long = i % 4 === 0 ? 1 : i % 2 === 0 ? 0.72 : 0.5;
      this.base[i] = long * (0.85 + rnd() * 0.3);
      this.thr[i] = rnd();
    }

    const s = Math.round(this.size * 0.5);
    this.coreWhite = this.makeCore(s, "255,255,255");
    this.coreRed = this.makeCore(s, "255,70,60");
    this.coreDeep = this.makeCore(s, "110,60,230");

    this.lightMq.addEventListener("change", this.onScheme);
    this.raf = requestAnimationFrame(this.frame);
  }

  private makeCore(px: number, rgb: string): HTMLCanvasElement {
    const c = document.createElement("canvas");
    c.width = c.height = px;
    const g = c.getContext("2d");
    if (g) {
      const r = px / 2;
      const gr = g.createRadialGradient(r, r, 0, r, r, r);
      gr.addColorStop(0, `rgba(${rgb},1)`);
      gr.addColorStop(0.12, `rgba(${rgb},0.85)`);
      gr.addColorStop(0.35, `rgba(${rgb},0.25)`);
      gr.addColorStop(1, `rgba(${rgb},0)`);
      g.fillStyle = gr;
      g.fillRect(0, 0, px, px);
    }
    return c;
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
    this.destroyed = true;
    cancelAnimationFrame(this.raf);
    this.lightMq.removeEventListener("change", this.onScheme);
  }

  private frame = (ts: number): void => {
    if (this.destroyed) return;
    this.raf = requestAnimationFrame(this.frame);
    const t = ts / 1000;
    const dt = this.lastT < 0 ? 0.016 : Math.min(0.1, Math.max(0, t - this.lastT));
    this.lastT = t;
    this.draw(dt);
  };

  private draw(dt: number): void {
    const { ctx, size } = this;
    const motion = this.reduced.matches ? 0.25 : 1;
    const tgt = TARGETS[this.state];
    const cur = this.cur;

    // ease every parameter toward the state target
    const k = 1 - Math.exp(-dt * (this.reduced.matches ? 2 : 4));
    cur.active += (tgt.active - cur.active) * k;
    cur.bright += (tgt.bright - cur.bright) * k;
    cur.sat += (tgt.sat - cur.sat) * k;
    cur.light += (tgt.light - cur.light) * k;
    cur.len += (tgt.len - cur.len) * k;
    cur.audio += (tgt.audio - cur.audio) * k;
    cur.noise += (tgt.noise - cur.noise) * k;
    cur.spin += (tgt.spin - cur.spin) * k;
    cur.wave += (tgt.wave - cur.wave) * k;
    cur.err += (tgt.err - cur.err) * k;
    cur.flicker += (tgt.flicker - cur.flicker) * k;
    cur.breath += (tgt.breath - cur.breath) * k;

    // level: fast attack, slower release (frame-rate independent)
    const lk = 1 - Math.exp(-dt * (this.target > this.level ? 22 : 6));
    this.level += (this.target - this.level) * lk;
    const lv = this.level * (this.reduced.matches ? 0.5 : 1);

    this.tm += dt * motion;
    const tm = this.tm;
    this.rot += dt * cur.spin * motion;

    const light = this.light;
    const cx = size / 2;
    const cy = size / 2;
    const maxLen = size * 0.49;
    const inner = 0;

    // error flicker (slow, irregular)
    const flick = 1 - cur.flicker * (0.5 + 0.5 * Math.sin(tm * 3.1) * Math.sin(tm * 1.7 + 1));
    const breath = 1 + cur.breath * Math.sin(tm * 1.5);
    const hueDrift = tm * 9;
    const waveHead = tm * 2.4;

    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.globalCompositeOperation = "source-over";
    ctx.globalAlpha = 1;
    ctx.clearRect(0, 0, size, size);
    // additive glow on dark; plain alpha on light (additive is invisible on white)
    ctx.globalCompositeOperation = light ? "source-over" : "lighter";

    const satL = cur.sat;
    const lig = light ? cur.light * 0.72 : cur.light;
    const baseLen = cur.len * size * breath;
    const spreadAudio = cur.audio * lv;
    const glowA = light ? 0.16 : 0.13;
    const coreA = light ? 0.75 : 0.9;

    for (let pass = 0; pass < 2; pass++) {
      const wide = pass === 0;
      const halfW = wide ? 4.2 : 1.15;
      for (let i = 0; i < RAYS; i++) {
        // visibility for the sparse "connecting" state
        const vis = clamp01((cur.active - this.thr[i] * 0.8) * 6);
        if (vis <= 0.002) continue;

        const a = this.ang[i] + this.rot;
        const wob =
          Math.sin(tm * this.w1[i] + this.ph1[i]) * 0.6 +
          Math.sin(tm * this.w2[i] + this.ph2[i]) * 0.4;
        let len = baseLen * this.base[i] * (1 + cur.noise * wob);
        // audio stretch, each ray responds a bit differently
        len += size * 0.2 * spreadAudio * (0.55 + 0.45 * Math.sin(tm * 8 + this.ph1[i])) * (0.5 + this.base[i] * 0.6);
        if (len > maxLen) len = maxLen;
        if (len < size * 0.02) len = size * 0.02;

        let bright = cur.bright * vis * flick;
        if (cur.wave > 0.01) {
          // brightness wave running around the star
          const d = Math.cos(a - waveHead);
          const w = d > 0 ? d * d * d : 0;
          bright *= 1 - cur.wave * 0.55 + cur.wave * 0.9 * w;
          len *= 1 + cur.wave * 0.18 * w;
        }
        if (bright > 1) bright = 1;

        // hue: rainbow, blended toward red on error
        let h = (this.hue[i] + hueDrift * motion + 360) % 360;
        if (cur.err > 0.001) {
          const red = h > 180 ? 360 : 0;
          h += (red - h) * cur.err;
        }
        ctx.fillStyle = `hsla(${h.toFixed(0)},${satL.toFixed(0)}%,${(wide ? lig : Math.min(92, lig + 14)).toFixed(0)}%,${(bright * (wide ? glowA * 3 : coreA)).toFixed(3)})`;

        const cs = Math.cos(a);
        const sn = Math.sin(a);
        const x0 = cx + cs * inner;
        const y0 = cy + sn * inner;
        const x1 = cx + cs * len;
        const y1 = cy + sn * len;
        // thin triangle: wide at the core, needle point at the tip
        ctx.beginPath();
        ctx.moveTo(x0 - sn * halfW, y0 + cs * halfW);
        ctx.lineTo(x1, y1);
        ctx.lineTo(x0 + sn * halfW, y0 - cs * halfW);
        ctx.closePath();
        ctx.fill();
      }
    }

    // bright core + halo (pre-rendered sprites, no gradient allocation)
    let coreScale = size * (0.2 + 0.08 * spreadAudio) * (0.85 + 0.15 * breath);
    const ca = clamp01(cur.bright * 1.15) * flick;
    ctx.globalCompositeOperation = "source-over";
    const drawCore = (img: HTMLCanvasElement, alpha: number): void => {
      if (alpha <= 0.003) return;
      ctx.globalAlpha = clamp01(alpha);
      ctx.drawImage(img, cx - coreScale, cy - coreScale, coreScale * 2, coreScale * 2);
    };
    drawCore(this.coreRed, ca * cur.err);
    if (light) {
      drawCore(this.coreDeep, ca * 0.55 * (1 - cur.err));
    } else {
      ctx.globalCompositeOperation = "lighter";
      drawCore(this.coreWhite, ca * (1 - cur.err));
      coreScale *= 0.4;
      drawCore(this.coreWhite, ca * (1 - cur.err));
    }
    ctx.globalCompositeOperation = "source-over";
    ctx.globalAlpha = 1;
  }
}

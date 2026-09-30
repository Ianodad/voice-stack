"""
Web UI Task 3 check: FastAPI server (history REST + WebRTC signaling + session manager).

Starts the REAL runtime (models + mlx_lm.server, ~10-60s) and the real app on a random
port in-process, temp history DB, then asserts:
  - history REST CRUD + 404
  - /api/offer with a headless aiortc client returns an SDP answer (real negotiation:
    the client applies the answer and ICE connects)
  - two consecutive /api/offer calls leave exactly ONE live worker
  - client message type "interrupt" reaches the server and stops bot audio (headless
    audio check; needs the bot to actually speak a reply to spike_input.wav)

Also covers the assistant action routes (/api/actions/*), session-end discard of pending
actions, and the expired-card `actions_cleared` hook.

Run: uv run python scripts/check_web.py
  - If port 8080 is already served (the user's `voice-stack web`), the LLM server is REUSED
    and never stopped; the LLM-restart and MLXLMServer sections are skipped in that case.
  - `--stub`: no models / LLM at all; a fake worker replaces the pipeline. Runs history,
    guard, failed-start and the whole actions section (real aiortc negotiation).
"""

import asyncio
import json
import socket
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import uvicorn
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.processors.frameworks.rtvi import RTVIServerMessageFrame

import voice_stack.server as srv
from voice_stack import toolset
from voice_stack.actions import PendingActions
from voice_stack.history import History
from voice_stack.runtime import SPIKE_AUDIO_PATH, Runtime
from voice_stack.server import create_app


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Client:
    """Headless aiortc client: sends spike_input.wav, records bot audio energy."""

    def __init__(self, base: str, http: httpx.AsyncClient, wav: Path | None):
        self.base, self.http, self.wav = base, http, wav
        self.pc = RTCPeerConnection()
        self.dc = self.pc.createDataChannel("chat", ordered=True)
        self.audio_log: list[tuple[float, float]] = []  # (monotonic, rms)
        self.connected = asyncio.Event()
        self.pc_id = None
        self.conversation_id = None
        self._player = None

        @self.pc.on("connectionstatechange")
        async def _():
            if self.pc.connectionState == "connected":
                self.connected.set()

        @self.pc.on("track")
        def on_track(track):
            if track.kind == "audio":
                asyncio.ensure_future(self._drain(track))

    async def _drain(self, track):
        try:
            while True:
                frame = await track.recv()
                pcm = frame.to_ndarray().astype(np.float32)
                self.audio_log.append((time.monotonic(), float(np.sqrt(np.mean(pcm**2)))))
        except Exception:
            pass

    async def offer(self, query: str = ""):
        if self.wav is not None:
            self._player = MediaPlayer(str(self.wav))
            self.pc.addTrack(self._player.audio)
        else:
            self.pc.addTransceiver("audio", direction="sendrecv")
        await self.pc.setLocalDescription(await self.pc.createOffer())
        r = await self.http.post(
            f"{self.base}/api/offer{query}",
            json={"sdp": self.pc.localDescription.sdp, "type": self.pc.localDescription.type},
        )
        assert r.status_code == 200, (r.status_code, r.text)
        ans = r.json()
        assert ans["type"] == "answer" and ans["sdp"] and ans["pc_id"], ans
        self.pc_id = ans["pc_id"]
        self.conversation_id = r.headers.get("x-conversation-id")
        await self.pc.setRemoteDescription(RTCSessionDescription(ans["sdp"], ans["type"]))
        return ans

    def send_rtvi(self, t: str, d=None):
        self.dc.send(
            json.dumps({"label": "rtvi-ai", "type": "client-message", "id": "x1", "data": {"t": t, "d": d}})
        )

    def speaking_now(self, window: float = 0.3, thresh: float = 200.0) -> bool:
        now = time.monotonic()
        recent = [v for ts, v in self.audio_log if now - ts <= window]
        return bool(recent) and max(recent) > thresh

    async def close(self):
        await self.pc.close()


STUB = "--stub" in sys.argv


class FakeWorker:
    """Stands in for the Pipecat worker in --stub mode."""

    def __init__(self):
        self.stopped = asyncio.Event()
        self.frames: list = []
        self.rtvi = SimpleNamespace(event_handler=lambda name: (lambda fn: fn))

    async def queue_frame(self, f):
        self.frames.append(f)

    async def cancel(self, reason=None):
        self.stopped.set()


class FakeRunner:
    def __init__(self, *a, **k):
        self._w = None

    async def add_workers(self, w):
        self._w = w

    async def run(self):
        await self._w.stopped.wait()


def fake_build_worker(transport, runtime, messages, **k):
    return FakeWorker(), None


def make_root(d: str) -> Path:
    root = Path(d).resolve()
    (root / "archive").mkdir()
    (root / "notes").mkdir()
    for n in ("a.txt", "b.txt", "c.txt", "d.txt", "e.txt", "f.txt"):
        (root / "notes" / n).write_text(f"content of {n}\n")
    return root


def rec(worker) -> list:
    """Record every frame queued to `worker` (still forwarding to the real queue_frame)."""
    frames: list = []
    orig = worker.queue_frame

    async def spy(f, *a, **k):
        frames.append(f)
        return await orig(f, *a, **k)

    worker.queue_frame = spy
    return frames


def msgs(frames, typ=None) -> list[dict]:
    return [f.data for f in frames
            if isinstance(f, RTVIServerMessageFrame) and (typ is None or f.data.get("type") == typ)]


def spoken(frames) -> list[str]:
    return [f.text for f in frames if isinstance(f, TTSSpeakFrame)]


async def unit_sweep() -> None:
    """Model-free: an expired card swept silently by propose pushes actions_cleared BEFORE the new card."""
    with tempfile.TemporaryDirectory() as d:
        root = make_root(d)
        now = [0.0]
        pend = PendingActions(root, clock=lambda: now[0], ttl=300.0)
        session = toolset.ToolSession("S-sweep", root, pend)
        _, handlers = toolset.build(session)

        class L:
            def __init__(self): self.frames = []
            async def push_frame(self, f, *a, **k): self.frames.append(f)

        async def propose(src, dst, llm):
            out = {}

            async def cb(result, properties=None): out["r"] = result
            await handlers["move_file"](SimpleNamespace(
                function_name="move_file", tool_call_id="1", arguments={"src": src, "dst": dst},
                llm=llm, pipeline_worker=None, context=SimpleNamespace(messages=[]),
                result_callback=cb, app_resources=session))
            return out["r"]

        l1 = L()
        r = await propose("notes/a.txt", "archive/a.txt", l1)
        assert r.get("status") == "awaiting_user_confirmation", r
        assert [m["type"] for m in msgs(l1.frames)] == ["pending_action"], msgs(l1.frames)
        # card still live: second proposal is refused (429) and does NOT push actions_cleared
        l2 = L()
        r = await propose("notes/b.txt", "archive/b.txt", l2)
        assert "error" in r and not msgs(l2.frames, "actions_cleared"), (r, msgs(l2.frames))
        # expire it; the next proposal sweeps it silently -> actions_cleared then pending_action
        now[0] = 301.0
        l3 = L()
        r = await propose("notes/b.txt", "archive/b.txt", l3)
        assert r.get("status") == "awaiting_user_confirmation", r
        order = [m["type"] for m in msgs(l3.frames)]
        assert order == ["actions_cleared", "pending_action"], order
        # a fresh first-ever card (nothing shown before) gets no actions_cleared
        session2 = toolset.ToolSession("S-fresh", root, pend)
        _, h2 = toolset.build(session2)
        l4 = L()
        out = {}

        async def cb2(result, properties=None): out["r"] = result
        await h2["move_file"](SimpleNamespace(
            function_name="move_file", tool_call_id="1", arguments={"src": "notes/c.txt", "dst": "archive/c.txt"},
            llm=l4, pipeline_worker=None, context=SimpleNamespace(messages=[]),
            result_callback=cb2, app_resources=session2))
        assert [m["type"] for m in msgs(l4.frames)] == ["pending_action"], msgs(l4.frames)
    print("expired-card sweep pushes actions_cleared before the new card: ok")


async def actions_section(base, http, app, sessions, root: Path) -> None:
    pend = app.state.pending
    assert pend.root == root

    # --- no session: pending -> [], approve/deny -> 409 "no active session" (documented) ---
    assert sessions.current_session_id() is None
    r = await http.get(f"{base}/api/actions/pending")
    assert r.status_code == 200 and r.json() == [], (r.status_code, r.text)
    for verb in ("approve", "deny"):
        r = await http.post(f"{base}/api/actions/whatever/{verb}", json={})
        assert r.status_code == 409 and r.json()["detail"] == "no active session", (verb, r.status_code, r.text)

    # --- guard on the new routes ---
    for verb in ("approve", "deny"):
        u = f"{base}/api/actions/x/{verb}"
        assert (await http.post(u, json={}, headers={"Host": "evil.example.com"})).status_code == 403
        assert (await http.post(u, json={}, headers={"Origin": "https://evil.example.com"})).status_code == 403
        r = await http.post(u, content="{}", headers={"Content-Type": "text/plain"})
        assert r.status_code == 415, (verb, r.status_code)
    assert (await http.get(f"{base}/api/actions/pending", headers={"Host": "evil.example.com"})).status_code == 403
    assert (await http.get(f"{base}/api/actions/pending", headers={"Origin": "https://evil.example.com"})).status_code == 403
    print("actions: no-session + guard: ok")

    # --- live session ---
    c = Client(base, http, None)
    await c.offer()
    await asyncio.wait_for(c.connected.wait(), 20)
    sid = sessions.current_session_id()
    assert sid, "no live session id"
    frames = rec(sessions._current.worker)
    notes = root / "notes"
    arch = root / "archive"

    r = await http.get(f"{base}/api/actions/pending")
    assert r.status_code == 200 and r.json() == []
    for verb in ("approve", "deny"):
        r = await http.post(f"{base}/api/actions/nope/{verb}", json={})
        assert r.status_code == 404, (verb, r.status_code, r.text)
        assert "nope" not in r.text, "route echoed the raw id"

    # approve: executes, pushes action_result + speaks
    p = pend.propose(sid, "move", {"src": "notes/a.txt", "dst": "archive/a.txt"})
    lst = (await http.get(f"{base}/api/actions/pending")).json()
    assert [x["id"] for x in lst] == [p.id] and set(lst[0]) == {"id", "kind", "summary", "diff", "expires_in"}, lst
    r = await http.post(f"{base}/api/actions/{p.id}/approve", json={})
    assert r.status_code == 200 and r.json()["status"] == "done", (r.status_code, r.text)
    assert (arch / "a.txt").exists() and not (notes / "a.txt").exists(), "move did not happen"
    res = msgs(frames, "action_result")
    assert res and res[-1]["status"] == "done" and res[-1]["id"] == p.id and res[-1]["summary"], res
    assert any(t.startswith("Done.") for t in spoken(frames)), spoken(frames)
    assert any(f.append_to_context for f in frames if isinstance(f, TTSSpeakFrame) and f.text.startswith("Done."))
    r = await http.post(f"{base}/api/actions/{p.id}/approve", json={})
    assert r.status_code == 409, r.status_code                       # already used
    assert (await http.get(f"{base}/api/actions/pending")).json() == []
    assert msgs(frames, "action_result")[-1]["status"] == "done", "double-click must not push 'expired'"
    print("actions: approve executes once, pushes result + speaks, second approve 409: ok")

    # concurrent double approve: exactly one 200
    p = pend.propose(sid, "move", {"src": "notes/b.txt", "dst": "archive/b.txt"})
    rs = await asyncio.gather(*[http.post(f"{base}/api/actions/{p.id}/approve", json={}) for _ in range(6)])
    codes = sorted(x.status_code for x in rs)
    assert codes == [200] + [409] * 5, codes
    assert (arch / "b.txt").exists() and not (notes / "b.txt").exists()
    assert len([m for m in msgs(frames, "action_result") if m["id"] == p.id and m["status"] == "done"]) == 1
    print("actions: concurrent approve -> exactly one 200: ok")

    # deny
    p = pend.propose(sid, "move", {"src": "notes/c.txt", "dst": "archive/c.txt"})
    r = await http.post(f"{base}/api/actions/{p.id}/deny", json={})
    assert r.status_code == 200, (r.status_code, r.text)
    assert (notes / "c.txt").exists() and not (arch / "c.txt").exists(), "denied action ran"
    assert msgs(frames, "action_result")[-1] == {"type": "action_result", "id": p.id, "status": "denied"}
    assert "Okay, I won't." in spoken(frames)
    assert (await http.post(f"{base}/api/actions/{p.id}/deny", json={})).status_code == 409
    assert (await http.post(f"{base}/api/actions/{p.id}/approve", json={})).status_code == 409
    print("actions: deny ok, nothing executed: ok")

    # expiry (injected clock): approve -> 409, pushes expired + actions_cleared, file untouched
    now = [1000.0]
    real_clock = pend._clock
    pend._clock = lambda: now[0]
    try:
        p = pend.propose(sid, "move", {"src": "notes/d.txt", "dst": "archive/d.txt"})
        now[0] += 301
        n_before = len(frames)
        r = await http.post(f"{base}/api/actions/{p.id}/approve", json={})
        assert r.status_code == 409, r.status_code
        new = msgs(frames[n_before:])
        assert [m["type"] for m in new] == ["action_result", "actions_cleared"] and new[0]["status"] == "expired", new
        assert (notes / "d.txt").exists() and not (arch / "d.txt").exists()
        assert not [t for t in spoken(frames[n_before:]) if t.startswith("Done")]
    finally:
        pend._clock = real_clock
    print("actions: expired approve -> 409 + expired + actions_cleared: ok")

    # foreign session id: 404, and the other session's action stays pending
    other = pend.propose("someone-else", "move", {"src": "notes/e.txt", "dst": "archive/e.txt"})
    for verb in ("approve", "deny"):
        r = await http.post(f"{base}/api/actions/{other.id}/{verb}", json={})
        assert r.status_code == 404, (verb, r.status_code)
    assert [x["id"] for x in pend.list("someone-else")] == [other.id]
    assert (notes / "e.txt").exists()
    pend.discard_session("someone-else")
    print("actions: foreign id -> 404, untouched: ok")

    # session replacement discards pending; old id can no longer be approved
    p = pend.propose(sid, "move", {"src": "notes/f.txt", "dst": "archive/f.txt"})
    old_frames = frames
    c2 = Client(base, http, None)
    await c2.offer()
    await asyncio.wait_for(c2.connected.wait(), 20)
    sid2 = sessions.current_session_id()
    assert sid2 and sid2 != sid
    assert pend.list(sid) == [], "old session's pending survived replacement"
    assert (await http.get(f"{base}/api/actions/pending")).json() == [], "new session inherited pending"
    assert any(m["type"] == "actions_cleared" for m in msgs(old_frames)), "no actions_cleared on session end"
    for verb in ("approve", "deny"):
        r = await http.post(f"{base}/api/actions/{p.id}/{verb}", json={})
        assert r.status_code in (404, 409), (verb, r.status_code)
    assert (notes / "f.txt").exists() and not (arch / "f.txt").exists(), "old action executed after replacement"
    print("actions: session replacement discards pending, late approve refused: ok")

    # session ends (stop): pending discarded; approve with no session -> 409
    frames2 = rec(sessions._current.worker)
    p2 = pend.propose(sid2, "move", {"src": "notes/f.txt", "dst": "archive/f.txt"})
    await sessions.stop()
    assert pend.list(sid2) == [], "pending survived session stop"
    assert any(m["type"] == "actions_cleared" for m in msgs(frames2)), "no actions_cleared on stop"
    r = await http.post(f"{base}/api/actions/{p2.id}/approve", json={})
    assert r.status_code == 409 and r.json()["detail"] == "no active session", (r.status_code, r.text)
    assert (notes / "f.txt").exists()
    assert (await http.get(f"{base}/api/actions/pending")).json() == []
    await c.close()
    await c2.close()
    print("actions: stop discards pending; approve with no session 409: ok")

    # discarded-during-request race: action id taken after the session changed is refused
    # (covered by the replacement check above: captured sid mismatch -> PendingActions 404/409).


async def main() -> None:
    await unit_sweep()
    reuse = False
    if STUB:
        rt = SimpleNamespace(executor=None)
        srv.build_worker = fake_build_worker
        srv.WorkerRunner = FakeRunner
        print("STUB mode: fake worker, no models/LLM")
    else:
        rt = Runtime()
        try:
            socket.create_connection(("127.0.0.1", 8080), timeout=1).close()
            reuse = True
            print("LLM server already running on :8080 -> reusing it (will not start/stop/restart it)")
            rt._start_llm_server = lambda: None
        except OSError:
            pass
    with tempfile.TemporaryDirectory() as d:
        try:
            if not STUB:
                await rt.start()
            history = History(Path(d) / "h.db")
            (Path(d) / "sandbox").mkdir()
            root = make_root(str(Path(d) / "sandbox"))
            app = create_app(rt, history, None, root=root)
            port = free_port()
            server = uvicorn.Server(
                uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
            )
            serve = asyncio.create_task(server.serve())
            while not server.started:
                await asyncio.sleep(0.05)
            base = f"http://127.0.0.1:{port}"
            sessions = app.state.sessions

            async with httpx.AsyncClient(timeout=60) as http:
                # --- history REST ---
                assert (await http.get(f"{base}/api/conversations")).json() == []
                assert (await http.get(f"{base}/api/conversations/x")).status_code == 404
                cid = history.create()
                history.append(cid, "user", "hi")
                history.append(cid, "assistant", "hello")
                lst = (await http.get(f"{base}/api/conversations")).json()
                assert [c["id"] for c in lst] == [cid], lst
                msgs = (await http.get(f"{base}/api/conversations/{cid}")).json()
                assert [m["role"] for m in msgs] == ["user", "assistant"], msgs
                r = await http.delete(f"{base}/api/conversations/{cid}")
                assert r.status_code == 200
                assert (await http.get(f"{base}/api/conversations")).json() == []
                print("history REST: ok")

                # --- request guard (I2): Host / Origin / Content-Type ---
                r = await http.get(f"{base}/api/conversations", headers={"Host": "evil.example.com"})
                assert r.status_code == 403, f"bad Host accepted: {r.status_code}"
                r = await http.get(f"{base}/api/conversations", headers={"Origin": "http://evil.example.com"})
                assert r.status_code == 403, f"foreign Origin accepted: {r.status_code}"
                r = await http.post(f"{base}/api/llm/restart", headers={"Origin": "https://evil.example.com"})
                assert r.status_code == 403, f"cross-site restart accepted: {r.status_code}"
                r = await http.post(
                    f"{base}/api/offer", content='{"sdp":"x","type":"offer"}',
                    headers={"Content-Type": "text/plain"},
                )
                assert r.status_code == 415, f"text/plain offer accepted: {r.status_code}"
                r = await http.get(f"{base}/api/conversations", headers={"Origin": base})
                assert r.status_code == 200, "same-origin request rejected"
                assert "access-control-allow-origin" not in r.headers
                print("request guard: ok")

                # --- failed session start (I1): 503, no dangling pc, no orphan conversation ---
                real_build = srv.build_worker

                def boom(*a, **k):
                    raise RuntimeError("forced build failure")

                srv.build_worker = boom
                try:
                    pc = RTCPeerConnection()
                    pc.createDataChannel("chat")
                    pc.addTransceiver("audio", direction="sendrecv")
                    await pc.setLocalDescription(await pc.createOffer())
                    r = await http.post(
                        f"{base}/api/offer",
                        json={"sdp": pc.localDescription.sdp, "type": pc.localDescription.type},
                    )
                    await pc.close()
                finally:
                    srv.build_worker = real_build
                assert r.status_code == 503, (r.status_code, r.text)
                assert app.state.handler._pcs_map == {}, "dangling peer connection"
                assert (await http.get(f"{base}/api/conversations")).json() == [], "orphan conversation"
                assert sessions.live_workers() == 0
                print("failed start -> 503, no pc, no orphan conversation: ok")

                if not STUB:
                    # --- offer #1 (continues a fresh conversation), real negotiation ---
                    c1 = Client(base, http, None)
                    await c1.offer()
                    await asyncio.wait_for(c1.connected.wait(), 20)
                    assert c1.conversation_id, "missing X-Conversation-Id"
                    assert sessions.live_workers() == 1
                    print(f"offer #1: answer ok, ICE connected, conversation={c1.conversation_id}")

                    # --- offer #2 replaces #1: exactly one live worker ---
                    # (an empty conversation is pruned when its session ends, so give it a turn)
                    history.append(c1.conversation_id, "user", "hello there")
                    c2 = Client(base, http, None)
                    await c2.offer(f"?conversation_id={c1.conversation_id}")
                    await asyncio.wait_for(c2.connected.wait(), 20)
                    await asyncio.sleep(1.0)
                    live = sessions.live_workers()
                    assert live == 1, f"expected exactly 1 live worker after 2 offers, got {live}"
                    assert c2.conversation_id == c1.conversation_id
                    assert len(sessions._all_tasks) == 2
                    assert sessions._all_tasks[0].done() and not sessions._all_tasks[1].done()
                    print("two consecutive offers -> exactly one live worker: ok")
                    await c1.close()
                    await c2.close()

                    # --- interrupt: queue a long TTS utterance straight into the live worker
                    # (deterministic; no LLM), measure how long bot audio plays uninterrupted
                    # (control) vs. when the client sends the RTVI "interrupt" message. ---
                    c3 = Client(base, http, None)
                    await c3.offer()
                    await asyncio.wait_for(c3.connected.wait(), 20)
                    worker = sessions._current.worker
                    long_text = (
                        "This is a deliberately long test sentence so that the bot keeps talking. "
                        "It goes on and on about nothing in particular. " * 8
                    )

                    async def speech_seconds(interrupt_after: float | None) -> float:
                        c3.audio_log.clear()
                        await worker.queue_frame(TTSSpeakFrame(long_text))
                        t0 = time.monotonic()
                        while not c3.speaking_now() and time.monotonic() - t0 < 30:
                            await asyncio.sleep(0.02)
                        assert c3.speaking_now(), "bot never spoke"
                        t_start = time.monotonic()
                        if interrupt_after is not None:
                            await asyncio.sleep(interrupt_after)
                            assert c3.speaking_now(), "bot not speaking when interrupt sent"
                            c3.send_rtvi("interrupt")
                        while c3.speaking_now(window=1.5) and time.monotonic() - t_start < 60:
                            await asyncio.sleep(0.05)
                        return time.monotonic() - t_start

                    full = await speech_seconds(None)
                    await asyncio.sleep(1.0)
                    cut = await speech_seconds(2.0)
                    print(f"interrupt: uninterrupted={full:.1f}s, interrupted at 2.0s -> stopped at {cut:.1f}s")
                    assert full > 10, f"control utterance too short to be meaningful ({full:.1f}s)"
                    assert cut < 4.5, f"bot kept speaking after interrupt ({cut:.1f}s)"
                    await asyncio.sleep(1.0)
                    assert not c3.speaking_now(window=1.0), "bot resumed after interrupt"
                    print("interrupt: ok (headless audio measurement; not audible-by-human)")
                    assert sessions.live_workers() == 1, "c3 session should still be live"

                    if reuse:
                        print('LLM restart section SKIPPED (reusing the running server; never restarting it)')
                        await c3.close()
                    else:
                        # --- LLM restart: event loop stays responsive; old log handle closed ---
                        old_log = rt.llm_server._log_file
                        restart = asyncio.create_task(http.post(f"{base}/api/llm/restart", timeout=120))
                        worst = 0.0
                        while not restart.done():
                            t = time.monotonic()
                            assert (await http.get(f"{base}/api/conversations")).status_code == 200
                            worst = max(worst, time.monotonic() - t)
                            await asyncio.sleep(0.2)
                        assert (await restart).status_code == 200
                        assert worst < 1.0, f"event loop blocked during restart ({worst:.2f}s)"
                        assert old_log is not None and old_log.closed, "old llm log handle leaked"
                        assert sessions.live_workers() == 0, "restart left a live worker (live session)"
                        await c3.close()
                        print(f"llm restart WITH live session: ok (worst concurrent GET latency {worst*1000:.0f}ms)")

                # --- empty conversations are pruned; ones with messages are kept ---
                async def wait_no_workers():
                    for _ in range(100):
                        if sessions.live_workers() == 0:
                            return
                        await asyncio.sleep(0.1)
                    raise AssertionError("session did not end")

                kept = history.create()
                history.append(kept, "user", "keep me")
                c4 = Client(base, http, None)
                await c4.offer()
                await asyncio.wait_for(c4.connected.wait(), 20)
                live_cid = c4.conversation_id
                assert live_cid in {c["id"] for c in history.list()}, "live conversation missing"
                await c4.close()
                await asyncio.sleep(0.5)
                await sessions.stop()
                await wait_no_workers()
                ids = {c["id"] for c in history.list()}
                assert live_cid not in ids, "empty conversation left behind after disconnect"
                assert kept in ids, "conversation with messages was pruned"
                empties = [c for c in history.list() if not history.get(c["id"])]
                assert not empties, f"empty rows remain: {empties}"
                print("empty conversation pruned, non-empty kept: ok")

                await actions_section(base, http, app, sessions, root)

            if reuse or STUB:
                print('MLXLMServer section SKIPPED')
            else:
                # --- MLXLMServer: stop() then start() must not spawn (M1) ---
                import subprocess

                from voice_stack.llm_server import MLXLMServer
                from voice_stack.runtime import LLM_MODEL_ID

                def n_mlx() -> int:
                    out = subprocess.run(["pgrep", "-f", "mlx_lm.server"], capture_output=True, text=True)
                    return len(out.stdout.split())

                before = n_mlx()
                srv2 = MLXLMServer(model_id=LLM_MODEL_ID, port=8099)
                srv2.stop()
                for fn in (srv2.start, srv2.restart):
                    try:
                        fn(timeout=5)
                    except RuntimeError:
                        pass
                    else:
                        raise AssertionError("start/restart after stop() did not refuse")
                await asyncio.sleep(0.5)
                assert n_mlx() == before, "stopped MLXLMServer spawned a process"
                print("MLXLMServer refuses spawn after stop(): ok")

            server.should_exit = True
            await serve
            assert sessions.live_workers() == 0, "workers left after server shutdown"
            print("\ncheck_web: PASS" + (" (stub mode)" if STUB else " (reused running LLM; restart/MLX sections skipped)" if reuse else ""))
        finally:
            if not STUB:
                rt.stop()


asyncio.run(main())

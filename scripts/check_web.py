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

Run: uv run python scripts/check_web.py
"""

import asyncio
import json
import socket
import tempfile
import time
from pathlib import Path

import httpx
import numpy as np
import uvicorn
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer
from pipecat.frames.frames import TTSSpeakFrame

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


async def main() -> None:
    rt = Runtime()
    with tempfile.TemporaryDirectory() as d:
        try:
            await rt.start()
            history = History(Path(d) / "h.db")
            app = create_app(rt, history, None)
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

                # --- offer #1 (continues a fresh conversation), real negotiation ---
                c1 = Client(base, http, None)
                await c1.offer()
                await asyncio.wait_for(c1.connected.wait(), 20)
                assert c1.conversation_id, "missing X-Conversation-Id"
                assert sessions.live_workers() == 1
                print(f"offer #1: answer ok, ICE connected, conversation={c1.conversation_id}")

                # --- offer #2 replaces #1: exactly one live worker ---
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
                await c3.close()

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
                assert sessions.live_workers() == 0, "restart left a live worker"
                print(f"llm restart: ok (worst concurrent GET latency {worst*1000:.0f}ms)")

            server.should_exit = True
            await serve
            assert sessions.live_workers() == 0, "workers left after server shutdown"
            print("\ncheck_web: PASS")
        finally:
            rt.stop()


asyncio.run(main())

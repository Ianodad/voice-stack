"""FastAPI server: WebRTC signaling, history REST, single active voice session."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pipecat.frames.frames import InterruptionFrame
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.request_handler import (
    ConnectionMode,
    IceCandidate,
    SmallWebRTCPatchRequest,
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
)
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.workers.runner import WorkerRunner

from voice_stack.bot import build_worker
from voice_stack.history import History
from voice_stack.runtime import Runtime

CANCEL_TIMEOUT = 10.0


class _Session:
    def __init__(self, cid: str, connection, worker, runner: WorkerRunner):
        self.cid = cid
        self.connection = connection
        self.worker = worker
        self.runner = runner
        self.task: asyncio.Task | None = None


class SessionManager:
    """Owns the single live voice session. A new session replaces the old one."""

    def __init__(self, runtime: Runtime, history: History):
        self._runtime = runtime
        self._history = history
        self._lock = asyncio.Lock()
        self._current: _Session | None = None
        self._all_tasks: list[asyncio.Task] = []

    def live_workers(self) -> int:
        """Number of session tasks (ever started) that have not finished."""
        return sum(1 for t in self._all_tasks if not t.done())

    async def start(self, connection, conversation_id: str | None) -> str:
        async with self._lock:
            await self._cancel_current("replaced")

            known = {c["id"] for c in self._history.list()}
            cid = conversation_id if conversation_id in known else self._history.create()
            history = self._history
            transport = SmallWebRTCTransport(
                connection,
                TransportParams(audio_in_enabled=True, audio_out_enabled=True),
            )
            worker, _ = build_worker(
                transport,
                self._runtime,
                history.context_window(cid),
                mute_while_bot_speaks=False,
                on_turn=lambda role, content: history.append(cid, role, content),
            )
            runner = WorkerRunner(handle_sigint=False, handle_sigterm=False)
            session = _Session(cid, connection, worker, runner)

            @worker.rtvi.event_handler("on_client_message")
            async def _on_client_message(rtvi, msg):
                if msg.type == "interrupt":
                    await worker.queue_frame(InterruptionFrame())

            @transport.event_handler("on_client_disconnected")
            async def _on_disconnected(transport, webrtc_connection):
                # No lock here: start() holds it while awaiting the old session,
                # and that session's own disconnect event lands in this handler.
                if self._current is session:
                    asyncio.create_task(self._cancel_session(session, "client disconnected"))

            async def _run() -> None:
                try:
                    await runner.add_workers(worker)
                    await runner.run()
                except Exception:
                    logger.exception("voice session crashed")
                finally:
                    if self._current is session:
                        self._current = None

            session.task = asyncio.create_task(_run())
            self._all_tasks.append(session.task)
            self._current = session
            return cid

    async def stop(self) -> None:
        async with self._lock:
            await self._cancel_current("stopped")

    async def _cancel_current(self, reason: str) -> None:
        session = self._current
        if session is not None:
            await self._cancel_session(session, reason)
        self._current = None

    async def _cancel_session(self, session: _Session, reason: str) -> None:
        task = session.task
        if task is None or task.done():
            return
        try:
            await session.worker.cancel(reason=reason)
        except Exception:
            logger.exception("worker.cancel failed")
        done, _ = await asyncio.wait({task}, timeout=CANCEL_TIMEOUT)
        if not done:
            logger.warning("session did not stop in {}s; cancelling task", CANCEL_TIMEOUT)
            task.cancel()
            await asyncio.wait({task}, timeout=CANCEL_TIMEOUT)


def create_app(runtime: Runtime, history: History, static_dir: Path | None) -> FastAPI:
    handler = SmallWebRTCRequestHandler(connection_mode=ConnectionMode.SINGLE)
    sessions = SessionManager(runtime, history)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            yield
        finally:
            await sessions.stop()
            await handler.close()

    app = FastAPI(lifespan=lifespan)
    offer_lock = asyncio.Lock()
    app.state.sessions = sessions
    app.state.handler = handler

    @app.post("/api/offer")
    async def offer(request: Request, conversation_id: str | None = None):
        body = await request.json()
        req = SmallWebRTCRequest.from_dict(body)
        cids: list[str] = []
        async with offer_lock:
            # A fresh offer (no pc_id) replaces any live session; SINGLE mode
            # would otherwise reject it while the old peer connection exists.
            if not req.pc_id:
                await sessions.stop()
                await handler.close()

            async def on_connection(conn) -> None:
                cids.append(await sessions.start(conn, conversation_id))

            answer = await handler.handle_web_request(req, on_connection)
        if answer is None:
            raise HTTPException(status_code=500, detail="no SDP answer")
        headers = {"X-Conversation-Id": cids[0]} if cids else {}
        return JSONResponse(answer, headers=headers)

    @app.patch("/api/offer")
    async def patch_offer(request: Request):
        body = await request.json()
        candidates = [
            IceCandidate(
                candidate=c["candidate"],
                sdp_mid=c["sdp_mid"] if "sdp_mid" in c else c.get("sdpMid"),
                sdp_mline_index=(
                    c["sdp_mline_index"] if "sdp_mline_index" in c else c.get("sdpMLineIndex")
                ),
            )
            for c in body.get("candidates", [])
        ]
        await handler.handle_patch_request(
            SmallWebRTCPatchRequest(pc_id=body["pc_id"], candidates=candidates)
        )
        return {"status": "success"}

    @app.get("/api/conversations")
    def list_conversations():
        return history.list()

    @app.get("/api/conversations/{cid}")
    def get_conversation(cid: str):
        if cid not in {c["id"] for c in history.list()}:
            raise HTTPException(status_code=404, detail="conversation not found")
        return history.get(cid)

    @app.delete("/api/conversations/{cid}")
    def delete_conversation(cid: str):
        history.delete(cid)
        return {"status": "deleted"}

    @app.post("/api/llm/restart")
    async def restart_llm():
        async with offer_lock:
            await sessions.stop()
            await handler.close()
            try:
                await runtime.restart_llm()
            except Exception as e:
                logger.exception("LLM restart failed")
                raise HTTPException(status_code=500, detail=f"LLM restart failed: {e}")
        return {"status": "restarted"}

    if static_dir is not None and Path(static_dir).is_dir():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
    return app

"""FastAPI server: WebRTC signaling, history REST, single active voice session."""

from __future__ import annotations

import asyncio
import os
import re
import stat
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pipecat.frames.frames import InterruptionFrame, TTSSpeakFrame
from pipecat.processors.frameworks.rtvi import RTVIServerMessageFrame
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

from voice_stack import tools
from voice_stack.actions import ActionError, PendingActions
from voice_stack.bot import build_worker
from voice_stack.history import History
from voice_stack.runtime import Runtime
from voice_stack.toolset import (
    SPEAK_DENIED,
    SPEAK_DONE,
    SPEAK_DONE_DEFAULT,
    SPEAK_EXPIRED,
    SPEAK_FAILED,
    ToolSession,
    retire_cards,
)

CANCEL_TIMEOUT = 10.0
PUSH_TIMEOUT = 2.0

_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f\u200b-\u200f\u2028-\u202e\u2066-\u2069]")


def _clean(text: object, limit: int) -> str:
    """Strip control/bidi characters and cap the length (text may contain file names)."""
    return _CONTROL.sub("", str(text))[:limit].strip()


class _Session:
    def __init__(self, cid: str, connection, worker, runner: WorkerRunner,
                 tool_session: ToolSession | None = None):
        self.cid = cid
        self.tool_session = tool_session
        self.connection = connection
        self.worker = worker
        self.runner = runner
        self.task: asyncio.Task | None = None


class SessionManager:
    """Owns the single live voice session. A new session replaces the old one."""

    def __init__(self, runtime: Runtime, history: History,
                 pending: PendingActions | None = None, root: Path | None = None):
        self._runtime = runtime
        self._history = history
        self._pending = pending
        self._root = Path(root) if root is not None else tools.DEFAULT_ROOT
        self._lock = asyncio.Lock()
        self._current: _Session | None = None
        self._all_tasks: list[asyncio.Task] = []
        self._bg: set[asyncio.Task] = set()  # strong refs to fire-and-forget cancels

    def live_workers(self) -> int:
        """Number of session tasks (ever started) that have not finished."""
        return sum(1 for t in self._all_tasks if not t.done())

    def current_session_id(self) -> str | None:
        """Tool-session id of the live session, or None when there is none."""
        s = self._current
        if s is None or s.tool_session is None or s.task is None or s.task.done():
            return None
        return s.tool_session.session_id

    def live_session(self, session_id: str) -> _Session | None:
        """The live session if (and only if) it still has this tool-session id."""
        s = self._current
        if s is not None and s.tool_session is not None and s.tool_session.session_id == session_id \
                and s.task is not None and not s.task.done():
            return s
        return None

    async def start(self, connection, conversation_id: str | None) -> str:
        async with self._lock:
            await self._cancel_current("replaced")
            self._history.prune_empty()

            known = {c["id"] for c in self._history.list()}
            created = conversation_id not in known
            cid = self._history.create() if created else conversation_id
            try:
                return await self._start_locked(connection, cid)
            except BaseException:
                # Don't leave an empty conversation behind for a failed start.
                if created and not self._history.get(cid):
                    self._history.delete(cid)
                raise

    async def _start_locked(self, connection, cid: str) -> str:
        history = self._history
        tool_session = (
            ToolSession(uuid.uuid4().hex, self._root, self._pending)
            if self._pending is not None else None
        )
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
            tools=tool_session,
        )
        runner = WorkerRunner(handle_sigint=False, handle_sigterm=False)
        session = _Session(cid, connection, worker, runner, tool_session)

        @worker.rtvi.event_handler("on_client_message")
        async def _on_client_message(rtvi, msg):
            if msg.type == "interrupt":
                await worker.queue_frame(InterruptionFrame())

        @transport.event_handler("on_client_disconnected")
        async def _on_disconnected(transport, webrtc_connection):
            # No lock here: start() holds it while awaiting the old session,
            # and that session's own disconnect event lands in this handler.
            if self._current is session:
                t = asyncio.create_task(self._cancel_session(session, "client disconnected"))
                self._all_tasks.append(t)
                self._bg.add(t)
                t.add_done_callback(self._bg.discard)

        async def _run() -> None:
            try:
                await runner.add_workers(worker)
                await runner.run()
            except Exception:
                logger.exception("voice session crashed")
            finally:
                if self._current is session:
                    self._current = None
                await self._discard(session)
                # Session over: drop conversations nobody spoke in, but never
                # the one belonging to a different, currently live session.
                live = self._current
                history.prune_empty(live.cid if live is not None else None)

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

    async def _discard(self, session: _Session) -> int:
        """Drop this session's pending actions (off the event loop). Never raises."""
        ts = session.tool_session
        if ts is None or self._pending is None:
            return 0
        try:
            return await asyncio.to_thread(self._pending.discard_session, ts.session_id)
        except Exception:
            logger.exception("discard_session failed")
            return 0

    async def push(self, session: _Session, data: dict, speak: str | None = None) -> None:
        """Best-effort RTVI message (and optional spoken line) to a live session's worker."""
        try:
            async def _go() -> None:
                await session.worker.queue_frame(RTVIServerMessageFrame(data=data))
                if speak:
                    await session.worker.queue_frame(TTSSpeakFrame(speak, append_to_context=True))
            await asyncio.wait_for(_go(), PUSH_TIMEOUT)
        except Exception:
            logger.warning("push {} to worker failed", data.get("type"))

    async def _cancel_session(self, session: _Session, reason: str) -> None:
        # Pending actions die with the session, even if the worker already ended.
        await self._discard(session)
        if session.tool_session is not None:
            session.tool_session.shown_id = None
        task = session.task
        if task is None or task.done():
            return
        await self.push(session, {"type": "actions_cleared"})
        try:
            await session.worker.cancel(reason=reason)
        except Exception:
            logger.exception("worker.cancel failed")
        done, _ = await asyncio.wait({task}, timeout=CANCEL_TIMEOUT)
        if not done:
            logger.warning("session did not stop in {}s; cancelling task", CANCEL_TIMEOUT)
            task.cancel()
            await asyncio.wait({task}, timeout=CANCEL_TIMEOUT)


def _secure_root(root: Path) -> None:
    """Create the sandbox root 0700. Only chmod a directory we just created, or an existing
    real (non-symlink) directory owned by this user; never follow a symlink."""
    if root.is_symlink():
        logger.warning("sandbox root {} is a symlink; leaving its permissions alone", root)
        if not root.is_dir():  # dangling link, or link to a file
            raise RuntimeError(f"{root} exists but is not a directory")
        return
    if os.path.lexists(root) and not root.is_dir():
        raise RuntimeError(f"{root} exists but is not a directory")
    created = not root.exists()
    try:
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
    except FileExistsError:
        raise RuntimeError(f"{root} exists but is not a directory") from None
    try:
        st = os.lstat(root)
        if created or (stat.S_ISDIR(st.st_mode) and st.st_uid == os.getuid()):
            if not created and stat.S_IMODE(st.st_mode) != 0o700:
                logger.info("sandbox root {}: mode {:o} -> 700", root, stat.S_IMODE(st.st_mode))
            os.chmod(root, 0o700)
        else:
            logger.warning("sandbox root {} is not a directory owned by this user; permissions unchanged", root)
    except OSError:
        logger.warning("could not chmod sandbox root {} to 0700", root)


def create_app(runtime: Runtime, history: History, static_dir: Path | None,
               root: Path | None = None) -> FastAPI:
    root = Path(root) if root is not None else tools.DEFAULT_ROOT
    handler = SmallWebRTCRequestHandler(connection_mode=ConnectionMode.SINGLE)
    pending = PendingActions(root)
    sessions = SessionManager(runtime, history, pending, root)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        _secure_root(root)
        try:
            yield
        finally:
            await sessions.stop()
            await handler.close()

    app = FastAPI(lifespan=lifespan)

    allowed_names = {"127.0.0.1", "localhost"}
    dev_origin = "http://localhost:5173" if os.environ.get("VOICE_STACK_DEV") else None

    @app.middleware("http")
    async def guard(request: Request, call_next):
        # Local-only app: defend against DNS rebinding (Host) and cross-site
        # requests (Origin, simple text/plain POSTs). No CORS headers are sent.
        host = request.headers.get("host", "")
        if host.rsplit(":", 1)[0] not in allowed_names:
            return JSONResponse({"detail": "bad Host"}, status_code=403)
        origin = request.headers.get("origin")
        if origin is not None and origin != dev_origin and origin not in (
            f"http://{host}",
        ):
            return JSONResponse({"detail": "bad Origin"}, status_code=403)
        if (
            request.url.path.startswith("/api/")
            and request.method in ("POST", "PATCH", "PUT", "DELETE")
            and (
                request.url.path.startswith("/api/actions/") or request.url.path == "/api/offer"
                or request.headers.get("content-length", "0") != "0"   # other routes: body present
            )
            and request.headers.get("content-type", "").split(";")[0].strip().lower()
            != "application/json"
        ):
            return JSONResponse({"detail": "Content-Type must be application/json"}, status_code=415)
        return await call_next(request)
    offer_lock = asyncio.Lock()
    app.state.sessions = sessions
    app.state.handler = handler
    app.state.pending = pending

    # ---- assistant actions -------------------------------------------------
    # With no live session: GET pending -> [] ; approve/deny -> 409 "no active session".
    _MESSAGES = {400: "bad request", 404: "no such pending action",
                 409: "that request expired or was already used", 429: "busy"}
    # Spoken lines go into the model's context as its own words: FIXED text only,
    # never file names or any model/user/tool text.
    _SPEAK_DONE = SPEAK_DONE
    _SPEAK_FAILED = SPEAK_FAILED

    def _error(e: ActionError) -> JSONResponse:
        if e.status == 422:
            return JSONResponse({"detail": _clean(e, 300) or "could not complete the action"}, status_code=422)
        if e.status in _MESSAGES:
            body = {"detail": _MESSAGES[e.status]}
            if e.reason == "used":
                body["error"] = "already handled"
            return JSONResponse(body, status_code=e.status)
        return JSONResponse({"detail": "internal error"}, status_code=500)

    def _clear_shown(live: _Session, action_id: str) -> None:
        ts = live.tool_session
        if ts is not None and ts.shown_id == action_id:
            ts.shown_id = None
        if ts is not None and ts.shown_id is None:
            retire_cards(ts.context)   # no card left: scrub its stale 'on screen' traces

    @app.get("/api/actions/pending")
    async def pending_actions():
        sid = sessions.current_session_id()
        if sid is None:
            return []
        return await asyncio.to_thread(pending.list, sid)

    async def _on_failure(sid: str, action_id: str, e: ActionError) -> None:
        live = sessions.live_session(sid)         # id must still match the live session
        if live is None:
            return
        if e.reason == "expired":
            _clear_shown(live, action_id)
            await sessions.push(live, {"type": "action_result", "id": action_id, "status": "expired"},
                                speak=SPEAK_EXPIRED)
            await sessions.push(live, {"type": "actions_cleared"})
        elif e.reason == "failed":
            _clear_shown(live, action_id)
            await sessions.push(
                live,
                {"type": "action_result", "id": action_id, "status": "failed",
                 "summary": "That did not work. Nothing was changed."},
                speak=_SPEAK_FAILED,
            )

    @app.post("/api/actions/{action_id}/approve")
    async def approve_action(action_id: str):
        sid = sessions.current_session_id()     # captured at request time
        if sid is None:
            return JSONResponse({"detail": "no active session"}, status_code=409)
        try:
            result = await asyncio.to_thread(pending.approve, action_id, sid)
        except ActionError as e:
            await _on_failure(sid, action_id, e)
            return _error(e)
        # Plan summary as-is: already escaped for display and identical to what the card shows.
        # (Never truncate it here: a cut can split a \\uXXXX / \\UXXXXXXXX / \\\\ escape.)
        summary = str(result.get("summary", ""))
        live = sessions.live_session(sid)
        if live is not None:
            _clear_shown(live, action_id)
            await sessions.push(
                live,
                {"type": "action_result", "id": action_id, "status": "done", "summary": summary},
                speak=_SPEAK_DONE.get(result.get("kind", ""), SPEAK_DONE_DEFAULT),
            )
        return {"status": "done", "id": action_id, "summary": summary}

    @app.post("/api/actions/{action_id}/deny")
    async def deny_action(action_id: str):
        sid = sessions.current_session_id()
        if sid is None:
            return JSONResponse({"detail": "no active session"}, status_code=409)
        try:
            await asyncio.to_thread(pending.deny, action_id, sid)
        except ActionError as e:
            await _on_failure(sid, action_id, e)
            return _error(e)
        live = sessions.live_session(sid)
        if live is not None:
            _clear_shown(live, action_id)
            await sessions.push(
                live, {"type": "action_result", "id": action_id, "status": "denied"},
                speak=SPEAK_DENIED,
            )
        return {"status": "denied", "id": action_id}

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

            errors: list[BaseException] = []

            async def on_connection(conn) -> None:
                # Pipecat logs and swallows callback errors; capture ours.
                try:
                    cids.append(await sessions.start(conn, conversation_id))
                except Exception as e:
                    logger.exception("session start failed")
                    errors.append(e)
                    raise

            answer = await handler.handle_web_request(req, on_connection)
            if errors:
                await handler.close()
                raise HTTPException(status_code=503, detail=f"session start failed: {errors[0]}")
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

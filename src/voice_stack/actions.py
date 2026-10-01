"""Pending actions: propose -> (human) approve/deny -> execute, with an audit log.

Security model:
- The plan is built server-side by `tools.plan_*` and stored as a private deep
  copy. `approve` takes only ids and applies ONLY that stored copy, never a
  caller-supplied dict. `Pending.public()` never exposes the raw plan.
- One pending action per session. Ids are single-use: `approve`/`deny` pop the
  id under a lock first, so concurrent approvals execute at most once.
- Ids are bound to the session that proposed them; another session sees 404.
- Approved actions execute strictly one at a time (separate execution lock); the
  state lock is never held while applying or while writing the audit log.
- Audit lines are ASCII-only JSON (control/bidi characters are escaped) and go
  to `root/.audit.jsonl`, which the file tools refuse to touch. An audit write
  failure is logged to stderr and never blocks or fails an action.
"""

from __future__ import annotations

import copy
import json
import os
import secrets
import stat
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

from . import tools
from .tools import ToolError

AUDIT_NAME = ".audit.jsonl"
SPENT_CAP = 500
MAX_ARG_CHARS = 2 * 1024 * 1024      # per string argument; tools enforce tighter limits
MAX_DETAIL_CHARS = 500

_KINDS: dict[str, tuple[str, ...]] = {
    "move": ("src", "dst"),
    "edit": ("path", "old_text", "new_text"),
}


class ActionError(Exception):
    """HTTP-shaped failure: 400 bad request, 404 unknown/wrong session,
    409 expired or already used, 422 could not execute, 429 one pending at a time."""

    def __init__(self, status: int, message: str = "", reason: str = ""):
        super().__init__(message or f"action error {status}")
        self.status = status
        # machine-readable: expired | used | unknown | wrong_session | busy | bad_request | failed
        self.reason = reason or {400: "bad_request", 404: "unknown", 409: "used",
                                 422: "failed", 429: "busy"}.get(status, "")


@dataclass
class Pending:
    id: str
    session_id: str
    kind: str
    plan: dict
    summary: str
    diff: str | None
    created_at: float
    expires_in: float = 0.0

    def public(self) -> dict:
        """Safe view for clients. `expires_in` is a snapshot taken when the copy was made."""
        return {"id": self.id, "kind": self.kind, "summary": self.summary,
                "diff": self.diff, "expires_in": max(0, int(round(self.expires_in)))}


class PendingActions:
    def __init__(self, root: Path, clock: Callable[[], float] = time.monotonic, ttl: float = 300.0):
        # `clock` drives expiry only (monotonic by default); audit stamps use wall time.
        self.root = Path(root)
        self._clock = clock
        self._ttl = float(ttl)
        self._lock = threading.Lock()          # state lock: short, no I/O
        self._exec_lock = threading.Lock()     # serialises apply_* calls
        self._audit_lock = threading.Lock()
        self._items: dict[str, Pending] = {}
        self._spent: OrderedDict[str, tuple[str, str]] = OrderedDict()   # id -> (owning session, how it ended)

    # ------------------------------------------------------------ audit
    def _audit(self, event: str, p: Pending | None = None, detail: str = "", *,
               id: str = "", kind: str = "") -> None:
        """Never raises, never blocks (O_NONBLOCK + regular-file check). Do not call
        while holding the state lock."""
        try:
            rec = {"t": round(time.time(), 3), "event": event,
                   "id": p.id if p else id, "kind": p.kind if p else kind,
                   "summary": p.summary if p else "", "detail": str(detail)[:MAX_DETAIL_CHARS]}
            line = json.dumps(rec, ensure_ascii=True) + "\n"      # escapes control/bidi chars
            with self._audit_lock:
                fd = os.open(self.root / AUDIT_NAME,
                             os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NONBLOCK
                             | getattr(os, "O_NOFOLLOW", 0), 0o600)
                try:
                    if not stat.S_ISREG(os.fstat(fd).st_mode):
                        raise OSError("audit path is not a regular file")
                    os.write(fd, line.encode("ascii"))
                finally:
                    os.close(fd)
        except Exception as e:                                     # never block the action
            print(f"audit write failed: {type(e).__name__}: {e}", file=sys.stderr)

    def _flush(self, events: list) -> None:
        for ev, p, detail in events:
            self._audit(ev, p, detail)

    # ------------------------------------------------------------ internals
    def _expired(self, p: Pending) -> bool:
        return self._clock() - p.created_at > self._ttl

    def _mark_spent(self, p: Pending, why: str = "used") -> None:
        """why: 'expired' (timed out) or 'used' (approved/denied/discarded)."""
        self._spent[p.id] = (p.session_id, why)
        while len(self._spent) > SPENT_CAP:
            self._spent.popitem(last=False)

    def _sweep_locked(self, events: list) -> None:
        for pid in [i for i, p in self._items.items() if self._expired(p)]:
            p = self._items.pop(pid)
            self._mark_spent(p, "expired")
            events.append(("expire", p, "expired before a decision"))

    def _view(self, p: Pending) -> Pending:
        """Caller-facing copy: mutating it cannot affect the stored plan."""
        left = self._ttl - (self._clock() - p.created_at)
        return replace(p, plan=copy.deepcopy(p.plan), expires_in=left)

    def _take(self, action_id: str, session_id: str, verb: str) -> Pending:
        """Pop the pending action for approve/deny, or raise. The decision is made
        under the state lock so concurrent callers cannot both win."""
        events: list = []
        try:
            with self._lock:
                p = self._items.get(action_id) if isinstance(action_id, str) else None
                if p is None:
                    owner = self._spent.get(action_id) if isinstance(action_id, str) else None
                    if owner is not None and owner[0] == session_id:
                        if owner[1] == "expired":
                            # swept earlier (GET/list or a new proposal); the first decision attempt reports it
                            self._spent[action_id] = (owner[0], "used")
                            raise ActionError(409, "that action expired", "expired")
                        raise ActionError(409, "that action was already used or has expired", "used")
                    raise ActionError(404, "no such pending action")
                if p.session_id != session_id:
                    raise ActionError(404, "no such pending action", "wrong_session")      # pending stays put
                del self._items[action_id]
                self._mark_spent(p)
                if self._expired(p):
                    events.append(("expire", p, f"expired before {verb}"))
                    raise ActionError(409, "that action expired", "expired")
                return p
        finally:
            self._flush(events)

    # ------------------------------------------------------------ API
    def propose(self, session_id: str, kind: str, args: dict) -> Pending:
        if not isinstance(session_id, str) or not session_id:
            raise ActionError(400, "bad session")
        if kind not in _KINDS:
            raise ActionError(400, "unknown action kind")
        if not isinstance(args, dict):
            raise ActionError(400, "args must be an object")
        keys = _KINDS[kind]
        if set(args) != set(keys):
            raise ActionError(400, f"{kind} needs exactly: {', '.join(keys)}")
        for k in keys:
            if not isinstance(args[k], str):
                raise ActionError(400, f"{k} must be text")
            if len(args[k]) > MAX_ARG_CHARS:
                raise ActionError(400, f"{k} is too large")
        self._check_slot(session_id)
        # plan outside the state lock (fuzzy lookup can be slow)
        if kind == "move":
            plan = tools.plan_move(self.root, args["src"], args["dst"])
        else:
            plan = tools.plan_edit(self.root, args["path"], args["old_text"], args["new_text"])
        events: list = []
        try:
            with self._lock:
                self._sweep_locked(events)
                if any(p.session_id == session_id for p in self._items.values()):
                    raise ActionError(429, "confirm or deny the card on screen first")
                pid = secrets.token_urlsafe(8)
                while pid in self._items or pid in self._spent:
                    pid = secrets.token_urlsafe(8)
                p = Pending(id=pid, session_id=session_id, kind=kind, plan=copy.deepcopy(plan),
                            summary=str(plan.get("summary", "")), diff=plan.get("diff"),
                            created_at=self._clock())
                self._items[pid] = p
                events.append(("propose", p, p.diff or ""))
                view = self._view(p)
        finally:
            self._flush(events)
        return view

    def _check_slot(self, session_id: str) -> None:
        events: list = []
        try:
            with self._lock:
                self._sweep_locked(events)
                if any(p.session_id == session_id for p in self._items.values()):
                    raise ActionError(429, "confirm or deny the card on screen first")
        finally:
            self._flush(events)

    def approve(self, action_id: str, session_id: str) -> dict:
        p = self._take(action_id, session_id, "approval")
        self._audit("approve", p)
        try:
            with self._exec_lock:       # one approved action at a time; state lock NOT held
                result = tools.apply_move(self.root, copy.deepcopy(p.plan)) if p.kind == "move" \
                    else tools.apply_edit(self.root, copy.deepcopy(p.plan))
        except ToolError as e:
            self._audit("execute_fail", p, str(e))
            raise ActionError(422, str(e)) from e
        except Exception as e:
            self._audit("execute_fail", p, f"{type(e).__name__}: {e}")
            raise ActionError(422, "the action could not be completed; nothing was changed") from e
        self._audit("execute_ok", p, json.dumps(result if isinstance(result, dict) else {}, default=str))
        return {**(result if isinstance(result, dict) else {}), "status": "done", "summary": p.summary,
                "kind": p.kind}

    def deny(self, action_id: str, session_id: str) -> None:
        p = self._take(action_id, session_id, "denial")
        self._audit("deny", p)

    def discard_session(self, session_id: str) -> int:
        events: list = []
        with self._lock:
            gone = [p for p in self._items.values() if p.session_id == session_id]
            for p in gone:
                del self._items[p.id]
                self._mark_spent(p)
                events.append(("discard", p, "session ended"))
        self._flush(events)
        return len(gone)

    def list(self, session_id: str) -> list[dict]:
        events: list = []
        with self._lock:
            self._sweep_locked(events)
            out = [self._view(p).public() for p in self._items.values() if p.session_id == session_id]
        self._flush(events)
        return out

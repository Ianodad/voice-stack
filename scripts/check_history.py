"""
Web UI Task 1 check: SQLite conversation history (voice_stack.history).

Run: uv run python scripts/check_history.py
"""

import tempfile
from pathlib import Path

from voice_stack.history import History

with tempfile.TemporaryDirectory() as d:
    h = History(Path(d) / "h.db")
    cid = h.create()
    for i in range(13):
        h.append(cid, "user", f"u{i}")
        h.append(cid, "assistant", f"a{i}")
    h.append(cid, "assistant", "   ")          # empty ignored
    assert len(h.get(cid)) == 26
    w = h.context_window(cid, 20)
    assert len(w) <= 20 and w[0]["role"] == "user" and w[-1]["content"] == "a12"
    assert h.list()[0]["title"] == "u0"
    assert h.get("nope") == []
    h.delete(cid)
    assert h.list() == [] and h.get(cid) == []

    # trailing-user case: a conversation with ONLY one user message
    cid2 = h.create()
    h.append(cid2, "user", "only one")
    w2 = h.context_window(cid2)
    assert w2 == [{"role": "user", "content": "only one"}], w2

    print("check_history.py: PASS")

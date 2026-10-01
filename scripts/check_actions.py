import tempfile, threading, json, os, time, inspect
from pathlib import Path
from voice_stack.actions import PendingActions, ActionError
from voice_stack import tools as T

class Clock:
    t = 1000.0
    def __call__(self): return self.t

def status(fn, *a):
    try: fn(*a)
    except ActionError as e: return e.status
    raise AssertionError("no ActionError")

with tempfile.TemporaryDirectory() as d:
    root = Path(d).resolve(); (root/"a.txt").write_text("hello\n"); (root/"archive").mkdir()
    clk = Clock(); pa = PendingActions(root, clock=clk)
    # propose does NOT touch the file
    p = pa.propose("S1", "move", {"src": "a.txt", "dst": "archive/a.txt"})
    assert (root/"a.txt").exists() and p.public()["summary"]
    # wrong session -> 404, then right session executes once, second approve 409
    assert status(pa.approve, p.id, "S2") == 404
    assert pa.approve(p.id, "S1")["status"] == "done" and (root/"archive/a.txt").exists()
    assert status(pa.approve, p.id, "S1") == 409
    # deny
    (root/"b.txt").write_text("x"); q = pa.propose("S1", "move", {"src": "b.txt", "dst": "archive/b.txt"})
    pa.deny(q.id, "S1"); assert (root/"b.txt").exists() and status(pa.approve, q.id, "S1") == 409
    # expiry
    r = pa.propose("S1", "move", {"src": "b.txt", "dst": "archive/b2.txt"}); clk.t += 301
    assert status(pa.approve, r.id, "S1") == 409 and pa.list("S1") == [] and (root/"b.txt").exists()
    # exactly one pending per session: second proposal refused (429), a different session is independent
    (root/"m0.txt").write_text("x"); (root/"m1.txt").write_text("x")
    first = pa.propose("S1", "move", {"src": "m0.txt", "dst": "archive/m0.txt"})
    assert status(pa.propose, "S1", "move", {"src": "m1.txt", "dst": "archive/m1.txt"}) == 429
    assert pa.propose("S9", "move", {"src": "m1.txt", "dst": "archive/m1x.txt"}); pa.discard_session("S9")
    # after deny the slot frees up
    pa.deny(first.id, "S1"); pa.propose("S1", "move", {"src": "m1.txt", "dst": "archive/m1.txt"})
    # discard on session end
    assert pa.discard_session("S1") == 1 and pa.list("S1") == []
    # concurrent double-approve executes exactly once
    (root/"c.txt").write_text("c"); s = pa.propose("S3", "move", {"src": "c.txt", "dst": "archive/c.txt"}); res = []
    def go():
        try: res.append(pa.approve(s.id, "S3")["status"])
        except ActionError as e: res.append(e.status)
    ts = [threading.Thread(target=go) for _ in range(6)]; [t.start() for t in ts]; [t.join() for t in ts]
    assert res.count("done") == 1 and res.count(409) == 5, res
    # stale edit: file changes after proposal -> approve fails (not 'done'), file untouched
    (root/"e.txt").write_text("one two\n"); e = pa.propose("S4", "edit", {"path": "e.txt", "old_text": "one", "new_text": "1"})
    (root/"e.txt").write_text("changed\n")
    try: pa.approve(e.id, "S4"); raise AssertionError("stale edit executed")
    except (ActionError, T.ToolError): pass
    assert (root/"e.txt").read_text() == "changed\n"
    # audit log exists, is JSONL, records events, and is unreadable through tools
    lines = [json.loads(l) for l in (root/".audit.jsonl").read_text().splitlines()]
    assert {"propose", "approve", "deny", "expire", "execute_ok"} <= {l["event"] for l in lines}
    # ---- hardening extras
    # public form: exact keys, no raw plan, no session id
    (root/"h.txt").write_text("h"); h = pa.propose("S5", "move", {"src": "h.txt", "dst": "archive/h.txt"})
    pub = h.public(); assert set(pub) == {"id", "kind", "summary", "diff", "expires_in"}, pub
    assert "S5" not in json.dumps(pub) and pub["expires_in"] == 300
    clk.t += 100; assert pa.list("S5")[0]["expires_in"] == 200
    # stored plan is a deep copy: mutating the returned Pending cannot redirect approval
    h.plan["dst"] = "archive/EVIL.txt"; h.plan["src"] = "a.txt"
    assert pa.approve(h.id, "S5")["status"] == "done"
    assert (root/"archive/h.txt").exists() and not (root/"archive/EVIL.txt").exists()
    # wrong-session deny leaves the pending in place; spent id + wrong session stays 404
    (root/"i.txt").write_text("i"); i = pa.propose("S6", "move", {"src": "i.txt", "dst": "archive/i.txt"})
    assert status(pa.deny, i.id, "S7") == 404 and len(pa.list("S6")) == 1
    assert status(pa.approve, "nope", "S6") == 404
    pa.deny(i.id, "S6"); assert status(pa.approve, i.id, "S7") == 404 and status(pa.deny, i.id, "S6") == 409
    # bad kind / missing keys / wrong types / extra keys / huge args -> 400, never KeyError
    assert status(pa.propose, "S8", "delete", {"path": "a"}) == 400
    assert status(pa.propose, "S8", "move", {"src": "x"}) == 400
    assert status(pa.propose, "S8", "edit", {"path": "e.txt"}) == 400
    assert status(pa.propose, "S8", "move", {"src": 1, "dst": "x"}) == 400
    assert status(pa.propose, "S8", "move", None) == 400
    assert status(pa.propose, "S8", "move", {"src": "e.txt", "dst": "x", "extra": "y"}) == 400
    assert status(pa.propose, "S8", "edit", {"path": "e.txt", "old_text": "x"*(3*1024*1024), "new_text": "y"}) == 400
    assert status(pa.propose, 5, "move", {"src": "a", "dst": "b"}) == 400
    assert pa.list("S8") == []
    # a ToolError at propose propagates and does not occupy the slot
    try: pa.propose("S8", "move", {"src": "../x", "dst": "y"}); raise AssertionError
    except T.ToolError: pass
    assert pa.list("S8") == []
    # destination appears between propose and approve -> 422, source untouched
    (root/"j.txt").write_text("j"); j = pa.propose("S8", "move", {"src": "j.txt", "dst": "archive/j.txt"})
    (root/"archive/j.txt").write_text("squatter")
    assert status(pa.approve, j.id, "S8") == 422 and (root/"j.txt").exists()
    assert (root/"archive/j.txt").read_text() == "squatter"
    assert status(pa.approve, j.id, "S8") == 409 and pa.list("S8") == []   # slot freed
    # approve after discard_session -> 409; racing approve vs discard never double-executes
    (root/"k.txt").write_text("k"); k = pa.propose("S10", "move", {"src": "k.txt", "dst": "archive/k.txt"})
    assert pa.discard_session("S10") == 1 and status(pa.approve, k.id, "S10") == 409 and (root/"k.txt").exists()
    # expired via list sweep audits expire and the id is 409, slot freed
    (root/"l.txt").write_text("l"); l = pa.propose("S11", "move", {"src": "l.txt", "dst": "archive/l.txt"})
    clk.t += 10_000; assert pa.list("S11") == [] and status(pa.approve, l.id, "S11") == 409
    pa.propose("S11", "move", {"src": "l.txt", "dst": "archive/l.txt"}); pa.discard_session("S11")
    # exact expiry boundary: still valid at ttl, expired just after
    (root/"n.txt").write_text("n"); n = pa.propose("S12", "move", {"src": "n.txt", "dst": "archive/n.txt"})
    clk.t += 300; assert len(pa.list("S12")) == 1
    clk.t += 0.5; assert status(pa.approve, n.id, "S12") == 409 and (root/"n.txt").exists()
    # spent-id set is bounded
    for _ in range(520):
        (root/"z.txt").write_text("z"); zz = pa.propose("SZ", "move", {"src": "z.txt", "dst": "archive/zz"})
        pa.deny(zz.id, "SZ")
    assert len(pa._spent) <= 500
    # audit: ascii-only single-line records, no raw control chars, no full file text
    raw = (root/".audit.jsonl").read_text()
    assert raw.isascii() and all(ord(c) >= 32 or c == "\n" for c in raw)
    for l in raw.splitlines():
        rec = json.loads(l); assert {"t", "event", "id", "kind", "summary", "detail"} <= set(rec), rec
    evs = {json.loads(l)["event"] for l in raw.splitlines()}
    assert {"execute_fail", "discard"} <= evs, evs
    # audit failure must not break approval (audit path is a directory -> write fails)
with tempfile.TemporaryDirectory() as d:
    root = Path(d).resolve(); (root/"a.txt").write_text("a"); (root/".audit.jsonl").mkdir()
    pa = PendingActions(root); p = pa.propose("S", "move", {"src": "a.txt", "dst": "sub/a.txt"})
    assert pa.approve(p.id, "S")["status"] == "done" and (root/"sub/a.txt").exists()
# forged plan dict cannot be smuggled in: approve takes only ids
import inspect; assert list(inspect.signature(PendingActions.approve).parameters) == ["self", "action_id", "session_id"]
# ---- fix round 1
# (I1) two sessions editing the same file concurrently: never "done" while losing an edit
with tempfile.TemporaryDirectory() as d:
    root = Path(d).resolve(); pa = PendingActions(root)
    for trial in range(15):
        (root/"shared.txt").write_text("one two\n")
        u1 = pa.propose("U1", "edit", {"path": "shared.txt", "old_text": "one", "new_text": "1"})
        u2 = pa.propose("U2", "edit", {"path": "shared.txt", "old_text": "two", "new_text": "2"})
        out = {}
        def go(sid, aid):
            try: out[sid] = pa.approve(aid, sid)["status"]
            except ActionError as e: out[sid] = e.status
        ts = [threading.Thread(target=go, args=a) for a in (("U1", u1.id), ("U2", u2.id))]
        [t.start() for t in ts]; [t.join() for t in ts]
        txt = (root/"shared.txt").read_text()
        if out["U1"] == "done" and out["U2"] == "done": assert txt == "1 2\n", (out, txt)
        else: assert 422 in out.values() and "done" in out.values(), (out, txt)
    assert inspect.signature(PendingActions.__init__).parameters["clock"].default is time.monotonic
# (I2a) N threads proposing for one session: exactly one wins, rest 429
with tempfile.TemporaryDirectory() as d:
    root = Path(d).resolve(); pa = PendingActions(root)
    for i in range(8): (root/f"f{i}.txt").write_text("x")
    res = []
    def prop(i):
        try: pa.propose("SS", "move", {"src": f"f{i}.txt", "dst": f"out/f{i}.txt"}); res.append("ok")
        except ActionError as e: res.append(e.status)
    ts = [threading.Thread(target=prop, args=(i,)) for i in range(8)]; [t.start() for t in ts]; [t.join() for t in ts]
    assert res.count("ok") == 1 and res.count(429) == 7 and len(pa.list("SS")) == 1, res
# (I2b) FIFO at .audit.jsonl: nothing blocks, approval still works
with tempfile.TemporaryDirectory() as d:
    root = Path(d).resolve(); os.mkfifo(root/".audit.jsonl"); (root/"a.txt").write_text("a")
    pa = PendingActions(root); box = {}
    def work():
        box["p"] = pa.propose("F", "move", {"src": "a.txt", "dst": "sub/a.txt"})
        box["l"] = pa.list("OTHER"); box["r"] = pa.approve(box["p"].id, "F")["status"]
    t = threading.Thread(target=work, daemon=True); t.start(); t.join(5)
    assert not t.is_alive() and box.get("r") == "done" and box["l"] == [], box
# (M1) expiry follows the injected clock, audit stamp is wall time
with tempfile.TemporaryDirectory() as d:
    root = Path(d).resolve(); (root/"a.txt").write_text("a"); clk = Clock(); pa = PendingActions(root, clock=clk)
    pa.propose("W", "move", {"src": "a.txt", "dst": "b.txt"})
    t0 = json.loads((root/".audit.jsonl").read_text().splitlines()[0])["t"]; assert abs(t0 - time.time()) < 60
# (M2) machine-readable ActionError.reason
with tempfile.TemporaryDirectory() as d:
    root = Path(d).resolve(); (root/"a.txt").write_text("a"); (root/"archive").mkdir()
    clk = Clock(); pa = PendingActions(root, clock=clk)
    def reason(fn, *a):
        try: fn(*a)
        except ActionError as e: return e.reason
        raise AssertionError("no ActionError")
    p = pa.propose("R1", "move", {"src": "a.txt", "dst": "archive/a.txt"})
    assert reason(pa.approve, "nope", "R1") == "unknown"
    assert reason(pa.approve, p.id, "R2") == "wrong_session"
    assert reason(pa.propose, "R1", "move", {"src": "a.txt", "dst": "archive/z.txt"}) == "busy"
    pa.approve(p.id, "R1")
    assert reason(pa.approve, p.id, "R1") == "used" and reason(pa.deny, p.id, "R1") == "used"
    (root/"b.txt").write_text("b"); q = pa.propose("R1", "move", {"src": "b.txt", "dst": "archive/b.txt"})
    clk.t += 301
    assert reason(pa.approve, q.id, "R1") == "expired"
    (root/"c.txt").write_text("c"); r = pa.propose("R1", "move", {"src": "c.txt", "dst": "archive/c.txt"})
    (root/"c.txt").unlink()
    assert reason(pa.approve, r.id, "R1") == "failed"
# (M4) tools hide the audit file in every spelling
with tempfile.TemporaryDirectory() as d:
    root = Path(d).resolve(); (root/"a.txt").write_text("a"); (root/".audit.jsonl").write_text("{}\n")
    for name in (".audit.jsonl", ".AUDIT.JSONL"):
        for fn in (lambda: T.read_file(root, name), lambda: T.file_info(root, name),
                   lambda: T.plan_move(root, name, "x.txt"), lambda: T.plan_move(root, "a.txt", name),
                   lambda: T.plan_edit(root, name, "{", "x")):
            try: fn(); raise AssertionError(f"tool reached {name}")
            except T.ToolError: pass
    assert ".audit.jsonl" not in [e["name"] for e in T.list_dir(root)]
    assert all("audit" not in x.lower() for x in T.find_file(root, ".audit.jsonl"))
print("check_actions.py: PASS")

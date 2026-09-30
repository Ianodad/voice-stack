"""
Assistant-tools Task 1 check: sandboxed file tools (voice_stack.tools).

Run: uv run python scripts/check_tools.py
"""

import os, tempfile
from pathlib import Path
from voice_stack import tools as T

def raises(fn, *a, **k):
    try: fn(*a, **k)
    except T.ToolError as e: return e
    raise AssertionError(f"{fn.__name__}{a} did not raise")

with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as outside:
    root = Path(d).resolve()
    (root/"notes").mkdir(); (root/"archive").mkdir()
    (root/"notes/Shopping List.txt").write_text("milk\neggs\n")
    (root/"budget_draft.txt").write_text("March total: 10\n")
    (root/"invoice.txt").write_text("Date: March\nDue: March\n")
    (Path(outside)/"secret.txt").write_text("nope")
    os.symlink(outside, root/"link_out")
    # 1 sandbox escapes
    for bad in ["../x", "/etc/passwd", f"{outside}/secret.txt", "link_out/secret.txt", ".backups", ".BACKUPS/x", ".audit.jsonl", "notes/../../x"]:
        raises(T.resolve, root, bad)
    # 2 fuzzy resolution
    assert T.resolve(root, "notes/shopping_list") == root/"notes/Shopping List.txt"
    assert T.resolve(root, "BUDGET DRAFT") == root/"budget_draft.txt"
    e = raises(T.resolve, root, "nonexistent"); assert isinstance(e.near, list)
    (root/"a1.txt").write_text("x"); (root/"a2.txt").write_text("y")
    e = raises(T.resolve, root, "a"); assert {"a1.txt","a2.txt"} <= set(e.near)
    # 3 listing / find hide internals
    (root/".backups").mkdir(); (root/".audit.jsonl").write_text("")
    names = {x["name"] for x in T.list_dir(root)}
    assert ".backups" not in names and ".audit.jsonl" not in names and "notes" in names
    assert T.find_file(root, "shopping") == ["notes/Shopping List.txt"]
    (root/".backups"/"20260101").mkdir(parents=True, exist_ok=True); (root/".backups/20260101/backup_note.txt").write_text("x")
    assert T.find_file(root, "backup") == [] and T.find_file(root, "audit") == []
    raises(T.plan_move, root, "notes", "archive/notes2")   # directories refused
    # 4 read: truncation + binary
    (root/"big.txt").write_text("a"*70000)
    r = T.read_file(root, "big.txt"); assert r["truncated"] and len(r["content"]) == 65536
    (root/"bin.dat").write_bytes(b"\x00\x01\x02")
    assert raises(T.read_file, root, "bin.dat").args[0] == "not_text"
    assert T.file_info(root, "invoice")["lines"] == 2
    assert T.file_info(root, "zzz")["exists"] is False
    # 5 move: no overwrite, parents created, escape refused
    p = T.plan_move(root, "budget draft", "archive/2026/budget_draft.txt"); T.apply_move(root, p)
    assert (root/"archive/2026/budget_draft.txt").exists() and not (root/"budget_draft.txt").exists()
    raises(T.plan_move, root, "invoice", "notes/Shopping List.txt")   # dst exists
    raises(T.plan_move, root, "invoice", "../out.txt")
    raises(T.plan_move, root, "invoice", ".backups/x")
    # 6 edit: 0 / 2 / 1 matches, backup, atomic, diff
    assert "0" in str(raises(T.plan_edit, root, "invoice", "April", "May"))
    assert "2" in str(raises(T.plan_edit, root, "invoice", "March", "April"))
    p = T.plan_edit(root, "invoice", "Date: March", "Date: April")
    assert "-Date: March" in p["diff"] and "+Date: April" in p["diff"]
    out = T.apply_edit(root, p)
    assert (root/"invoice.txt").read_text() == "Date: April\nDue: March\n"
    b = root/out["backup"]; assert b.read_text() == "Date: March\nDue: March\n" and ".backups" in b.parts
    # stale: file changed after planning -> refuse, file untouched
    p2 = T.plan_edit(root, "invoice", "Due: March", "Due: May")
    (root/"invoice.txt").write_text("Due: March\nDue: March\n")
    raises(T.apply_edit, root, p2); assert (root/"invoice.txt").read_text() == "Due: March\nDue: March\n"

    # ---- extra adversarial asserts (beyond the brief) ----
    (root/"invoice.txt").write_text("Date: March\nDue: March\n")
    # 7 malformed paths
    for bad in ["", "a\x00b", "notes/\x00", "x" * 300, "invoice.txt/", "notes/Shopping List.txt/"]:
        raises(T.resolve, root, bad)
        raises(T.resolve, root, bad, must_exist=False)
    assert T.resolve(root, "notes/") == root/"notes"
    # 8 case / unicode variants of hidden names, and hidden via dot segments
    for bad in [".Backups", ".backups/20260101/backup_note.txt", ".BACKUPS/20260101/backup_note.txt",
                ".Audit.JSONL", "notes/../.backups/x", "./.backups", ".backups/", "archive/../.audit.jsonl",
                str(root/".backups"), str(root/".BACKUPS"/"x")]:
        raises(T.resolve, root, bad); raises(T.resolve, root, bad, must_exist=False)
    raises(T.read_file, root, ".BACKUPS/20260101/backup_note.txt")
    raises(T.file_info, root, ".audit.jsonl")
    raises(T.list_dir, root, ".backups"); raises(T.list_dir, root, ".BACKUPS")
    raises(T.plan_move, root, ".backups/20260101/backup_note.txt", "archive/leak.txt")
    raises(T.plan_move, root, "invoice", ".AUDIT.JSONL")
    raises(T.plan_edit, root, ".audit.jsonl", "a", "b")
    # 9 symlinks: parent dir out, dangling out, in-root alias to hidden, loops, file link out
    raises(T.resolve, root, "link_out/new.txt", must_exist=False)
    raises(T.plan_move, root, "invoice", "link_out/new.txt")
    raises(T.plan_move, root, "invoice", "link_out/sub/deeper/new.txt")
    os.symlink(Path(outside)/"not_there_yet", root/"dangle")
    raises(T.resolve, root, "dangle", must_exist=False); raises(T.plan_move, root, "invoice", "dangle")
    os.symlink(root/".backups", root/"sneak")
    raises(T.resolve, root, "sneak"); raises(T.resolve, root, "sneak/20260101/backup_note.txt")
    raises(T.read_file, root, "sneak/20260101/backup_note.txt"); raises(T.list_dir, root, "sneak")
    os.symlink(root/".audit.jsonl", root/"sneak2"); raises(T.read_file, root, "sneak2")
    os.symlink("loop_b", root/"loop_a"); os.symlink("loop_a", root/"loop_b")
    raises(T.resolve, root, "loop_a"); raises(T.resolve, root, "loop_a", must_exist=False)
    os.symlink(Path(outside)/"secret.txt", root/"innocent.txt")
    raises(T.read_file, root, "innocent.txt"); raises(T.plan_edit, root, "innocent.txt", "nope", "x")
    raises(T.plan_move, root, "innocent.txt", "archive/stolen.txt")
    # 10 listing/find never reveal escaping or hidden symlinks; fuzzy cannot walk through them
    names = {x["name"] for x in T.list_dir(root)}
    assert not names & {"link_out", "dangle", "sneak", "sneak2", "innocent.txt", "loop_a", ".backups"}, names
    assert T.find_file(root, "secret") == [], T.find_file(root, "secret")
    assert "innocent.txt" not in T.find_file(root, "innocent")
    bad_names = {"sneak", "sneak2", "link_out", "dangle", "loop_a", "loop_b", "innocent.txt"}
    for q in ("sneak", "20260101", "link", "loop", "dangle", "secret", "backup", "audit"):
        got = T.find_file(root, q)
        assert not [g for g in got if g.split("/")[0] in bad_names or ".backups" in g.lower() or "audit" in g.lower()], (q, got)
    raises(T.resolve, root, "link_out/secre"); raises(T.resolve, root, "link_out/secret")
    for q in ("zzz_none", "back", "audit", "sneak", "link"):
        near = raises(T.resolve, root, q).near
        assert not [x for x in near if x.split("/")[0] in bad_names or ".backups" in x.lower() or "audit" in x.lower()], (q, near)
    raises(T.file_info, root, "innocent.txt")      # link to outside: refused, no stat leak
    # 11 fuzzy never picks hidden / never guesses on ambiguity; unicode
    (root/"back_up_plan.txt").write_text("x")
    assert T.resolve(root, "back up plan") == root/"back_up_plan.txt"
    import unicodedata
    nfd = unicodedata.normalize("NFD", "café menu.txt"); (root/nfd).write_text("x")
    assert T.resolve(root, unicodedata.normalize("NFC", "café_menu")).exists()
    assert T.find_file(root, unicodedata.normalize("NFC", "café")) != []
    (root/"Report.txt").write_text("1"); (root/"report.md").write_text("2")
    assert raises(T.resolve, root, "report").args[0] == "ambiguous"
    # 12 apply_* re-validate and never fuzzy-resolve (stale / forged plans fail closed)
    (root/"m1.txt").write_text("one")
    pm = T.plan_move(root, "m1.txt", "archive/m1.txt")
    (root/"m1.txt").unlink(); (root/"m1.txt.bak").write_text("look-alike")
    raises(T.apply_move, root, pm)
    assert (root/"m1.txt.bak").read_text() == "look-alike" and not (root/"archive/m1.txt").exists()
    (root/"m2.txt").write_text("two"); pm = T.plan_move(root, "m2.txt", "archive/m2.txt")
    (root/"archive/m2.txt").write_text("already here")
    raises(T.apply_move, root, pm)
    assert (root/"archive/m2.txt").read_text() == "already here" and (root/"m2.txt").read_text() == "two"
    for forged in [
        {"kind": "move", "src": "m2.txt", "dst": "../escaped.txt"},
        {"kind": "move", "src": "m2.txt", "dst": ".backups/x.txt"},
        {"kind": "move", "src": "m2.txt", "dst": ".BACKUPS/x.txt"},
        {"kind": "move", "src": "m2.txt", "dst": "link_out/x.txt"},
        {"kind": "move", "src": "m2.txt", "dst": "dangle"},
        {"kind": "move", "src": ".audit.jsonl", "dst": "archive/a.txt"},
        {"kind": "move", "src": "../x", "dst": "archive/a.txt"},
        {"kind": "move", "src": "innocent.txt", "dst": "archive/a.txt"},
        {"kind": "edit", "src": "m2.txt", "dst": "archive/zz.txt"},
        {"kind": "move", "src": "m2.txt"},
        {},
    ]:
        raises(T.apply_move, root, forged)
    assert (root/"m2.txt").exists() and not (Path(d).parent/"escaped.txt").exists()
    assert list(Path(outside).iterdir()) == [Path(outside)/"secret.txt"]
    (root/"e1.txt").write_text("alpha beta\n")
    pe = T.plan_edit(root, "e1.txt", "alpha", "gamma")
    (root/"e1.txt").write_text("alpha beta\nTRAILING CHANGE\n")      # still one match, but content moved on
    raises(T.apply_edit, root, pe); assert (root/"e1.txt").read_text() == "alpha beta\nTRAILING CHANGE\n"
    (root/"e1.txt").write_text("alpha beta\n")
    pe = T.plan_edit(root, "e1.txt", "alpha", "gamma"); (root/"e1.txt").unlink()
    (root/"e1.txt.other").write_text("alpha")
    raises(T.apply_edit, root, pe)                                     # deleted -> no fuzzy to look-alike
    assert (root/"e1.txt.other").read_text() == "alpha" and not (root/"e1.txt").exists()
    (root/"e1.txt").write_text("alpha beta\n")
    for forged in [
        {"kind": "edit", "path": "../x", "old_text": "a", "new_text": "b"},
        {"kind": "edit", "path": ".backups/20260101/backup_note.txt", "old_text": "x", "new_text": "y"},
        {"kind": "edit", "path": ".BACKUPS/20260101/backup_note.txt", "old_text": "x", "new_text": "y"},
        {"kind": "edit", "path": "sneak/20260101/backup_note.txt", "old_text": "x", "new_text": "y"},
        {"kind": "edit", "path": "innocent.txt", "old_text": "nope", "new_text": "yes"},
        {"kind": "edit", "path": "e1.txt", "old_text": "alpha", "new_text": "gamma"},   # no content hash
        {"kind": "move", "path": "e1.txt", "old_text": "alpha", "new_text": "gamma"},
        {},
    ]:
        raises(T.apply_edit, root, forged)
    assert (root/"e1.txt").read_text() == "alpha beta\n"
    assert (root/".backups/20260101/backup_note.txt").read_text() == "x"
    assert (Path(outside)/"secret.txt").read_text() == "nope"
    # 13 edit details: mode kept, no temp litter, backups never collide, bad input
    os.chmod(root/"e1.txt", 0o640)
    before = set(os.listdir(root))
    o1 = T.apply_edit(root, T.plan_edit(root, "e1.txt", "alpha", "gamma"))
    o2 = T.apply_edit(root, T.plan_edit(root, "e1.txt", "gamma", "delta"))
    assert o1["backup"] != o2["backup"]
    assert (root/o1["backup"]).read_text() == "alpha beta\n" and (root/o2["backup"]).read_text() == "gamma beta\n"
    assert (root/"e1.txt").read_text() == "delta beta\n" and (os.stat(root/"e1.txt").st_mode & 0o777) == 0o640
    assert set(os.listdir(root)) == before, set(os.listdir(root)) ^ before
    raises(T.plan_edit, root, "e1.txt", "", "x")
    raises(T.plan_edit, root, "e1.txt", "delta", "a\x00b")
    assert raises(T.plan_edit, root, "bin.dat", "a", "b").args[0] == "not_text"
    (root/"huge.txt").write_text("a" * (1024 * 1024 + 1))
    assert raises(T.plan_edit, root, "huge.txt", "a", "b").args[0] == "too_large"
    (root/"latin.txt").write_bytes(b"caf\xe9")
    assert raises(T.read_file, root, "latin.txt").args[0] == "not_text"
    (root/"multi.txt").write_text("é" * 40000)                          # 80000 bytes; cut must not split a char
    r = T.read_file(root, "multi.txt"); assert r["truncated"] and set(r["content"]) == {"é"}
    raises(T.read_file, root, "notes")                                  # a directory is not a file
    # a symlink swapped in after planning: apply must not follow it out
    (root/"e2.txt").write_text("hello\n"); pe = T.plan_edit(root, "e2.txt", "hello", "bye")
    (root/"e2.txt").unlink(); os.symlink(Path(outside)/"secret.txt", root/"e2.txt")
    raises(T.apply_edit, root, pe); assert (Path(outside)/"secret.txt").read_text() == "nope"
    os.unlink(root/"e2.txt")
    # the move really is a move of one file, relative paths, no overwrite
    (root/"m3.txt").write_text("three"); pm = T.plan_move(root, "m3", "deep/er/m3.txt")
    assert pm["kind"] == "move" and pm["src"] == "m3.txt" and pm["dst"] == "deep/er/m3.txt"
    T.apply_move(root, pm); assert (root/"deep/er/m3.txt").read_text() == "three" and not (root/"m3.txt").exists()
    raises(T.apply_move, root, pm)                                      # replay: src gone
    # rel() is posix-relative
    assert T.rel(root, root/"deep/er/m3.txt") == "deep/er/m3.txt"
    # ---- fix round 1 ----
    # I1 lone surrogate / unencodable text
    (root/"s1.txt").write_text("hello\n")
    for nt in ["bye\ud800", "\udfff"]:
        raises(T.plan_edit, root, "s1.txt", "hello", nt)
    raises(T.plan_edit, root, "s1.txt", "hel\ud800lo", "x")
    raises(T.apply_edit, root, {"kind": "edit", "path": "s1.txt", "old_text": "hello", "new_text": "bye\ud800",
                                "sha256": "0" * 64})
    assert (root/"s1.txt").read_text() == "hello\n"
    # I2 read_file limit validation
    assert T.read_file(root, "big.txt", limit=-2)["content"] == "" and T.read_file(root, "big.txt", limit=-5)["truncated"]
    assert len(T.read_file(root, "big.txt", limit=10**9)["content"]) == 65536
    assert T.read_file(root, "big.txt", limit="10")["content"] == "a" * 10
    for badlim in [None, "ten", 1.5, [], True]:
        raises(T.read_file, root, "big.txt", limit=badlim)
    # I3 spoofing characters in paths and edit text
    for ch in ["\u202e", "\u202a", "\u2066", "\u200b", "\u200f", "\ufeff", "\u2028", "\u2029", "\n", "\r", "\x0b", "\x0c", "\x85", "\t", "\x1b", "\x7f"]:
        raises(T.plan_move, root, "invoice", f"archive/a{ch}b.txt")
        raises(T.plan_move, root, f"inv{ch}oice", "archive/zz.txt")
        raises(T.resolve, root, f"archive/a{ch}b.txt", must_exist=False)
        raises(T.apply_move, root, {"kind": "move", "src": "invoice.txt", "dst": f"archive/a{ch}b.txt"})
        if ch not in "\t\n\r":
            raises(T.plan_edit, root, "s1.txt", "hello", f"x{ch}y")
            raises(T.plan_edit, root, "s1.txt", f"hel{ch}lo", "y")
    raises(T.plan_move, root, "invoice", "archive/\u202etxt.exe")
    assert T.plan_edit(root, "s1.txt", "hello", "a\tb\nc\r\nd")["kind"] == "edit"   # legit whitespace ok
    # diff lines mirror real file lines: \x0b \x0c \x85 \u2028 inside a file do not split lines
    (root/"s2.txt").write_text("a\x0bb\nc\u2028d\x85e\nTARGET\n", encoding="utf-8")
    pd = T.plan_edit(root, "s2.txt", "TARGET", "DONE")
    lines = pd["diff"].split("\n")
    assert "-TARGET" in lines and "+DONE" in lines
    assert not any(c in pd["diff"] for c in "\x0b\x0c\x85\u2028\u2029\u202e"), repr(pd["diff"])
    assert " a\\u000bb" in lines and " c\\u2028d\\u0085e" in lines, lines
    (root/"s3.txt").write_text("one\r\ntwo\r\n")
    pd = T.plan_edit(root, "s3.txt", "two", "2"); assert "\r" not in pd["diff"] and "-two" in pd["diff"].split("\n")
    # M4 unlink failure after hardlink: no duplicate dst, no created dirs, retry works
    (root/"ro_dir").mkdir(); (root/"ro_dir/mv.txt").write_text("keep")
    pm = T.plan_move(root, "ro_dir/mv.txt", "fresh/sub/mv.txt")
    os.chmod(root/"ro_dir", 0o500)
    try:
        if os.geteuid() != 0:
            raises(T.apply_move, root, pm)
            assert (root/"ro_dir/mv.txt").read_text() == "keep"
            assert not (root/"fresh").exists(), "created dirs must be rolled back"
    finally:
        os.chmod(root/"ro_dir", 0o700)
    T.apply_move(root, pm); assert (root/"fresh/sub/mv.txt").read_text() == "keep" and not (root/"ro_dir/mv.txt").exists()
    # M5 result size cap
    (root/"cap.txt").write_text("x" + "a" * (1024 * 1024 - 10))
    assert raises(T.plan_edit, root, "cap.txt", "x", "y" * 100).args[0] == "too_large"
    pc = T.plan_edit(root, "cap.txt", "x", "y" * 5)
    pc["new_text"] = "y" * 100                     # forged-larger plan must be refused at apply
    raises(T.apply_edit, root, pc); assert (root/"cap.txt").read_text().startswith("xaaa")
    # M6 read-only file refused
    (root/"ro.txt").write_text("frozen\n"); os.chmod(root/"ro.txt", 0o400)
    raises(T.plan_edit, root, "ro.txt", "frozen", "thawed")
    (root/"ro2.txt").write_text("frozen\n"); pr = T.plan_edit(root, "ro2.txt", "frozen", "thawed")
    os.chmod(root/"ro2.txt", 0o400); raises(T.apply_edit, root, pr)
    assert (root/"ro2.txt").read_text() == "frozen\n"
    # M8 non-str args raise ToolError everywhere
    for bad in [None, 5, b"x", ["a"], {"a": 1}]:
        raises(T.resolve, root, bad); raises(T.list_dir, root, bad); raises(T.read_file, root, bad)
        raises(T.file_info, root, bad); raises(T.plan_move, root, bad, "a"); raises(T.plan_move, root, "invoice", bad)
        raises(T.plan_edit, root, bad, "a", "b"); raises(T.plan_edit, root, "s1.txt", bad, "b")
        raises(T.plan_edit, root, "s1.txt", "hello", bad)
        assert T.find_file(root, bad) == []
    # ---- fix round 2 ----
    # N1 limit strings that isdigit() but are not ints
    for badlim in ["²", "³", "①", "", "  ", "1e3", "0x10", "9" * 5000]:
        raises(T.read_file, root, "big.txt", limit=badlim)
    assert T.read_file(root, "big.txt", limit=" 7 ")["content"] == "a" * 7
    # N2 the diff shows exactly what is written: a change in \r is never invisible
    def written(path, old, new):
        pe = T.plan_edit(root, path, old, new); T.apply_edit(root, pe); return pe
    for content, old, new in [("x\ny\n", "x\n", "x\r\n"), ("x\ny\n", "x", "x\r"), ("x\r\ny\n", "x\r\n", "x\n"),
                              ("x\r\ny\r\n", "y", "y\r"), ("x\r\ny\r\n", "x\r\ny", "x\ny"), ("a\nb", "b", "b\n"),
                              ("a\nb\n", "b\n", "b")]:
        (root/"crlf.txt").write_bytes(content.encode())
        pe = T.plan_edit(root, "crlf.txt", old, new)
        assert pe["diff"] != "", (content, old, new)
        body = [l for l in pe["diff"].split("\n") if l[:1] in "+-" and not l.startswith(("+++", "---"))]
        assert body and (any("\\u000d" in l for l in body) or "\n" in old + new), (content, old, new, pe["diff"])
        T.apply_edit(root, pe)
        assert (root/"crlf.txt").read_bytes() == content.replace(old, new, 1).encode()
    # consistent CRLF file + consistent CRLF edit: the \r is hidden (clean diff)
    (root/"crlf.txt").write_bytes(b"one\r\ntwo\r\n")
    pe = T.plan_edit(root, "crlf.txt", "two", "2"); assert "\\u000d" not in pe["diff"] and "-two" in pe["diff"].split("\n")
    # N3 blank-looking characters are escaped in diff lines and summaries, but still editable text
    sneaky = {"\u00a0": "\\u00a0", "\u3164": "\\u3164", "\u115f": "\\u115f", "\u2800": "\\u2800",
              "\ue000": "\\ue000", "\u2003": "\\u2003", "\U000f0000": "\\U000f0000"}
    for ch, esc in sneaky.items():
        (root/"sn.txt").write_text(f"keep\n{ch}old\n", encoding="utf-8")
        pe = T.plan_edit(root, "sn.txt", "old", f"new{ch}x")
        assert ch not in pe["diff"] and esc in pe["diff"], (esc, pe["diff"])
        T.apply_edit(root, pe); assert ch in (root/"sn.txt").read_text(encoding="utf-8")
    (root/"sn.txt").write_text("a\n\u0301combining\nx\u0301y\n", encoding="utf-8")   # Mn at line start vs inside a word
    pe = T.plan_edit(root, "sn.txt", "combining", "C")
    assert "\\u0301combining" in pe["diff"] and "\u0301" not in pe["diff"].replace("x\u0301y", "")
    (root/"sn.txt").write_text("keep\nx\u0301y\nold\n", encoding="utf-8")
    assert "x\u0301y" in T.plan_edit(root, "sn.txt", "old", "new")["diff"]   # mid-word mark is left alone
    for ch, esc in [("\u00a0", "\\u00a0"), ("\u3164", "\\u3164"), ("\u2800", "\\u2800"), ("\ue000", "\\ue000"), ("\u0301", "\\u0301")]:
        name = f"{ch}z.txt"
        (root/name).write_text("q", encoding="utf-8")
        pm = T.plan_move(root, name, f"archive/{ch}zz.txt")
        assert ch not in pm["summary"] and esc in pm["summary"], pm["summary"]
        assert pm["src"] == name and pm["dst"] == f"archive/{ch}zz.txt"      # raw paths kept for apply
        (root/name).unlink()
        (root/name).write_text("q2\n", encoding="utf-8")
        pe = T.plan_edit(root, name, "q2", "r")
        assert ch not in pe["summary"] and esc in pe["summary"] and ch not in pe["diff"], pe["summary"]
        (root/name).unlink()
    # Known accepted limitation: Cf (ZWJ/ZWNJ) rejected in paths and edit text
    raises(T.plan_edit, root, "s1.txt", "hello", "a\u200db")
    print("check_tools.py: PASS")

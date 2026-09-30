"""Sandboxed file tools for the voice assistant.

Every tool goes through one path gate (`resolve`). The sandbox root holds the
user's files plus two internals the model must never touch: `.backups/` and
`.audit.jsonl`. This module makes no network calls and has no delete function.

Mutations are two-phase: `plan_*` validates and describes, `apply_*` re-validates
from scratch (exact paths only, no fuzzy lookup) so a stale or forged plan
fails closed. Residual risk: a symlink swapped in by another local process
between the final check and the syscall (the assistant itself cannot create
symlinks); file opens use O_NOFOLLOW to shrink that window.
"""

from __future__ import annotations

import difflib
import hashlib
import os
import re
import shutil
import stat
import tempfile
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_ROOT = Path.home() / "VoiceAssistant"
HIDDEN = (".backups", ".audit.jsonl")

MAX_EDIT_BYTES = 1024 * 1024
MAX_WALK_ENTRIES = 20000
_NEAR_N = 5


class ToolError(Exception):
    """Model-readable error. `near` lists candidate relative paths, if any."""

    def __init__(self, message: str, near: list[str] | None = None, code: str = ""):
        super().__init__(message)
        self.near: list[str] = list(near or [])
        self.code = code or message


# ---------------------------------------------------------------- name helpers

def _fold(s: str) -> str:
    return unicodedata.normalize("NFKC", s).casefold()


# Control, format (bidi overrides, zero-width, BOM), line/paragraph separators, surrogates
# are never allowed in a path: they could spoof what the confirmation card shows.
_BAD_PATH_CATS = frozenset({"Cc", "Cf", "Zl", "Zp", "Cs"})

_HIDDEN_FOLDED = {_fold(h) for h in HIDDEN}


def _is_hidden_name(name: str) -> bool:
    return _fold(name) in _HIDDEN_FOLDED


def _norm(s: str) -> str:
    """Fuzzy key: NFC-ish, case-insensitive, spaces/underscores/hyphens collapsed."""
    s = unicodedata.normalize("NFKC", s).casefold()
    return re.sub(r"[\s_\-]+", " ", s).strip()


def _stem(name: str) -> str:
    return name.rsplit(".", 1)[0] if "." in name.lstrip(".") else name


# ---------------------------------------------------------------- core gate

def _root_real(root: Path) -> Path:
    return Path(os.path.realpath(root))


def _lexists(p: Path) -> bool:
    try:
        os.lstat(p)
        return True
    except (OSError, ValueError):
        return False


def _exists(p: Path) -> bool:
    try:
        os.stat(p)
        return True
    except (OSError, ValueError):
        return False


def _inside(root_real: Path, real: Path) -> bool:
    return real == root_real or root_real in real.parents


def _blocked(root_real: Path, real: Path) -> str | None:
    """Return a reason string if `real` (an already-realpath'd path) is off limits."""
    if not _inside(root_real, real):
        return "outside"
    parts = real.relative_to(root_real).parts
    if parts and _is_hidden_name(parts[0]):
        return "hidden"
    # Case/normalisation-insensitive filesystems: an alias of a hidden entry that
    # the name check missed is still the same inode.
    if parts:
        for h in HIDDEN:
            hp = root_real / h
            try:
                if os.path.samefile(hp, root_real / parts[0]):
                    return "hidden"
            except (OSError, ValueError):
                pass
    return None


def _deny(reason: str) -> ToolError:
    # Same wording for outside and hidden so nothing leaks about what exists.
    return ToolError("path not allowed: it is outside the assistant folder", code="not_allowed")


def _candidate(root_real: Path, path: str) -> Path:
    if not isinstance(path, str):
        raise ToolError("path must be text", code="bad_path")
    if path == "" or path.strip() == "":
        raise ToolError("path is empty", code="bad_path")
    if any(unicodedata.category(c) in _BAD_PATH_CATS for c in path):
        raise ToolError("path contains an invalid or invisible character", code="bad_path")
    if len(path) > 4096 or any(len(seg.encode("utf-8", "surrogatepass")) > 255 for seg in path.split("/")):
        raise ToolError("path too long", code="bad_path")
    try:
        return Path(os.path.realpath(root_real / path if not os.path.isabs(path) else path))
    except (OSError, ValueError, RuntimeError):
        raise ToolError("path could not be resolved", code="bad_path")


def _all_rel_paths(root: Path, limit: int = MAX_WALK_ENTRIES) -> list[str]:
    return [rel(root, p) for p in _walk(root, limit)]


def _walk(root: Path, limit: int = MAX_WALK_ENTRIES):
    """Yield real paths of files and dirs under root: prunes hidden top-level names,
    skips symlinks that escape or point at hidden entries, never follows links."""
    rr = _root_real(root)
    count = 0
    for dirpath, dirnames, filenames in os.walk(rr, followlinks=False):
        here = Path(dirpath)
        keep = []
        for name in sorted(dirnames) + sorted(filenames):
            full = here / name
            if here == rr and _is_hidden_name(name):
                continue
            if os.path.islink(full):
                tgt = Path(os.path.realpath(full))
                if _blocked(rr, tgt) or os.path.islink(tgt) or not _exists(tgt):
                    continue  # escaping, hidden, looping or dangling link
            count += 1
            if count > limit:
                return
            keep.append(name)
            yield full
        # prune descent into hidden / escaping dirs
        dirnames[:] = [n for n in sorted(dirnames)
                       if not (here == rr and _is_hidden_name(n)) and not os.path.islink(here / n)]


def _near(root: Path, name: str, parent: Path | None = None, n: int = _NEAR_N) -> list[str]:
    key = _norm(name)
    rr = _root_real(root)
    out: list[str] = []
    if parent is not None and _inside(rr, parent) and not _blocked(rr, parent) and parent.is_dir():
        try:
            for ent in sorted(os.listdir(parent)):
                p = parent / ent
                if parent == rr and _is_hidden_name(ent):
                    continue
                if os.path.islink(p) and _blocked(rr, Path(os.path.realpath(p))):
                    continue
                k = _norm(ent)
                if key and (k.startswith(key) or key in k or _norm(_stem(ent)).startswith(key)):
                    out.append(rel(root, p))
        except OSError:
            pass
    paths = _all_rel_paths(root)
    by_name: dict[str, list[str]] = {}
    for rp in paths:
        by_name.setdefault(rp.rsplit("/", 1)[-1], []).append(rp)
    for m in difflib.get_close_matches(name, list(by_name), n=n, cutoff=0.4):
        out.extend(by_name[m])
    seen, res = set(), []
    for x in out:
        if x not in seen:
            seen.add(x)
            res.append(x)
    return res[:n]


def _resolve(root: Path, path: str, *, must_exist: bool, fuzzy: bool) -> Path:
    rr = _root_real(root)
    real = _candidate(rr, path)
    bad = _blocked(rr, real)
    if bad:
        raise _deny(bad)
    if os.path.islink(real):  # realpath leaves a link only for loops: refuse
        raise ToolError("path could not be resolved", code="bad_path")
    if path.endswith("/") and _exists(real) and not real.is_dir():
        raise ToolError("not a directory", code="not_a_directory")
    if _exists(real) or not must_exist:
        return real
    # missing: optionally try a unique forgiving match in the same (validated) parent
    name = real.name
    parent = real.parent
    if fuzzy and _inside(rr, parent) and not _blocked(rr, parent) and parent.is_dir() and name:
        key = _norm(name)
        hits = []
        try:
            for ent in sorted(os.listdir(parent)):
                if parent == rr and _is_hidden_name(ent):
                    continue
                if _norm(ent) == key or _norm(_stem(ent)) == key:
                    hits.append(ent)
        except OSError:
            hits = []
        # every hit is re-validated (symlinks resolved, containment, hidden)
        ok = []
        for h in hits:
            cand = Path(os.path.realpath(parent / h))
            if not _blocked(rr, cand) and _exists(cand):
                ok.append(cand)
        if len(ok) == 1:
            return ok[0]
        if len(ok) > 1:
            raise ToolError("ambiguous", near=[rel(root, c) for c in ok][:_NEAR_N], code="ambiguous")
        if len(hits) > 0:
            raise ToolError("not found", near=_near(root, name, parent), code="not_found")
    raise ToolError("not found", near=_near(root, name, parent) if name else [], code="not_found")


def resolve(root: Path, path: str, *, must_exist: bool = True) -> Path:
    """Map a user/model path to an absolute real path inside `root`.

    Raises ToolError for escapes, hidden internals, and (when must_exist) missing
    or ambiguous names; `near` lists candidates.
    """
    return _resolve(root, path, must_exist=must_exist, fuzzy=True)


def _resolve_exact(root: Path, path: str, *, must_exist: bool = True) -> Path:
    """Like resolve but never guesses (used by apply_*)."""
    return _resolve(root, path, must_exist=must_exist, fuzzy=False)


def rel(root: Path, p: Path) -> str:
    return Path(p).relative_to(_root_real(root)).as_posix()


# ---------------------------------------------------------------- safe IO

def _open_regular(real: Path) -> int:
    """Open read-only without following a final symlink; refuse non-regular files."""
    try:
        fd = os.open(real, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError:
        raise ToolError("cannot read that file", code="unreadable")
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ToolError("not a regular file", code="not_a_file")
    except BaseException:
        os.close(fd)
        raise
    return fd


def _read_bytes(real: Path, limit: int) -> tuple[bytes, bool]:
    fd = _open_regular(real)
    with os.fdopen(fd, "rb") as f:
        data = f.read(limit + 1)
    return data[:limit], len(data) > limit


def _decode(data: bytes, truncated: bool) -> str:
    if b"\x00" in data[:8192]:
        raise ToolError("not_text")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as e:
        if truncated and e.start >= len(data) - 3:
            return data[: e.start].decode("utf-8")  # cut landed mid-character
        raise ToolError("not_text")


# ---------------------------------------------------------------- read tools

def list_dir(root: Path, path: str = ".") -> list[dict]:
    if not isinstance(path, str):
        raise ToolError("path must be text", code="bad_path")
    real = resolve(root, path if path.strip() else ".")
    if not real.is_dir():
        raise ToolError("not a directory", code="not_a_directory")
    rr = _root_real(root)
    out = []
    try:
        entries = sorted(os.listdir(real), key=str.lower)
    except OSError:
        raise ToolError("cannot list that folder", code="unreadable")
    for name in entries:
        if real == rr and _is_hidden_name(name):
            continue
        full = real / name
        target = Path(os.path.realpath(full))
        if _blocked(rr, target):
            continue  # symlink pointing outside or at internals: invisible
        try:
            st = os.stat(target)
        except OSError:
            continue
        out.append({"name": name, "type": "dir" if stat.S_ISDIR(st.st_mode) else "file",
                    "size": 0 if stat.S_ISDIR(st.st_mode) else st.st_size})
    return out


def find_file(root: Path, name: str) -> list[str]:
    if not isinstance(name, str) or not name.strip() or "\x00" in name:
        return []
    key = _norm(name)
    scored: list[tuple[float, str]] = []
    by_name: dict[str, list[str]] = {}
    for p in _walk(root):
        r = rel(root, p)
        base = p.name
        k = _norm(base)
        if key in k or key in _norm(r):
            scored.append((0.0 if k == key or _norm(_stem(base)) == key else 1.0, r))
        else:
            by_name.setdefault(base, []).append(r)
    scored.sort()
    res = [r for _, r in scored]
    if len(res) < _NEAR_N:
        for m in difflib.get_close_matches(name, list(by_name), n=_NEAR_N, cutoff=0.6):
            res.extend(by_name[m])
    return res[:_NEAR_N]


def read_file(root: Path, path: str, limit: int = 65536) -> dict:
    if isinstance(limit, str):
        try:
            limit = int(limit.strip())
        except ValueError:
            raise ToolError("limit must be a whole number", code="bad_args")
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise ToolError("limit must be a whole number", code="bad_args")
    limit = max(0, min(limit, 65536))
    real = resolve(root, path)
    if real.is_dir():
        raise ToolError("that is a folder, not a file", code="not_a_file")
    data, truncated = _read_bytes(real, limit)
    return {"path": rel(root, real), "content": _decode(data, truncated), "truncated": truncated}


def file_info(root: Path, path: str) -> dict:
    try:
        real = resolve(root, path)
    except ToolError as e:
        if e.code in ("not_found", "ambiguous"):
            return {"exists": False, "near": e.near}
        raise
    st = os.stat(real)
    is_dir = stat.S_ISDIR(st.st_mode)
    lines = None
    if not is_dir and stat.S_ISREG(st.st_mode) and st.st_size <= MAX_EDIT_BYTES:
        try:
            data, _ = _read_bytes(real, MAX_EDIT_BYTES)
            text = _decode(data, False)
            lines = text.count("\n") + (1 if text and not text.endswith("\n") else 0)
        except ToolError:
            lines = None
    return {"path": rel(root, real), "exists": True, "type": "dir" if is_dir else "file",
            "size": 0 if is_dir else st.st_size,
            "modified": datetime.fromtimestamp(st.st_mtime, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "lines": lines}


# ---------------------------------------------------------------- move

def _check_move(root: Path, src: str, dst: str, *, exact: bool) -> tuple[Path, Path]:
    if not isinstance(dst, str) or dst.strip() == "" or dst.endswith("/"):
        raise ToolError("destination must be a file path", code="bad_path")
    real_src = _resolve(root, src, must_exist=True, fuzzy=not exact)
    if real_src.is_dir() or not stat.S_ISREG(os.stat(real_src).st_mode):
        raise ToolError("only files can be moved, not folders", code="not_a_file")
    real_dst = _resolve(root, dst, must_exist=False, fuzzy=False)
    if real_dst == _root_real(root):
        raise ToolError("destination must be a file path", code="bad_path")
    if _exists(real_dst) or _lexists(real_dst):
        raise ToolError("destination already exists; moves never overwrite", code="exists")
    return real_src, real_dst


def plan_move(root: Path, src: str, dst: str) -> dict:
    real_src, real_dst = _check_move(root, src, dst, exact=False)
    s, d = rel(root, real_src), rel(root, real_dst)
    return {"kind": "move", "src": s, "dst": d, "summary": f"Move {_display(s)} to {_display(d)}"}


def apply_move(root: Path, plan: dict) -> dict:
    if not isinstance(plan, dict) or plan.get("kind") != "move" \
            or not isinstance(plan.get("src"), str) or not isinstance(plan.get("dst"), str):
        raise ToolError("invalid move plan", code="bad_plan")
    real_src, real_dst = _check_move(root, plan["src"], plan["dst"], exact=True)
    rr = _root_real(root)
    made: list[Path] = []
    linked = False
    try:
        anc = real_dst.parent
        while not _lexists(anc) and anc != rr:
            made.append(anc)
            anc = anc.parent
        real_dst.parent.mkdir(parents=True, exist_ok=True)
        # re-check after mkdir: parent must still be inside root and not hidden
        parent_real = Path(os.path.realpath(real_dst.parent))
        if _blocked(rr, parent_real) or parent_real != real_dst.parent:
            raise ToolError("path not allowed: it is outside the assistant folder", code="not_allowed")
        try:
            os.link(real_src, real_dst)           # atomic, fails if dst appeared meanwhile
        except FileExistsError:
            raise ToolError("destination already exists; moves never overwrite", code="exists")
        except OSError:
            if _lexists(real_dst):
                raise ToolError("destination already exists; moves never overwrite", code="exists")
            real_src.rename(real_dst)             # filesystem without hard links
        else:
            linked = True
            os.unlink(real_src)
    except (ToolError, OSError) as e:
        # roll back: no duplicate link, no empty dirs this call created
        if linked and _lexists(real_src):
            try:
                os.unlink(real_dst)
            except OSError:
                pass
        for dpath in made:          # deepest first
            try:
                os.rmdir(dpath)
            except OSError:
                pass
        if isinstance(e, ToolError):
            raise
        raise ToolError(f"move failed: {e.strerror or 'filesystem error'}", code="io_error")
    return {"src": rel(root, real_src), "dst": rel(root, real_dst)}


# ---------------------------------------------------------------- edit

def _load_text(real: Path) -> tuple[str, bytes]:
    if not stat.S_ISREG(os.stat(real).st_mode):
        raise ToolError("not a regular file", code="not_a_file")
    data, truncated = _read_bytes(real, MAX_EDIT_BYTES)
    if truncated:
        raise ToolError("too_large")
    return _decode(data, False), data


def _check_edit_text(old_text: str, new_text: str) -> None:
    if not isinstance(old_text, str) or not isinstance(new_text, str):
        raise ToolError("old_text and new_text must be text", code="bad_args")
    if old_text == "":
        raise ToolError("old_text must not be empty", code="bad_args")
    for t in (old_text, new_text):
        for c in t:
            cat = unicodedata.category(c)
            if cat in ("Cf", "Zl", "Zp", "Cs") or (cat == "Cc" and c not in "\t\n\r"):
                raise ToolError("text contains an invalid or invisible character", code="bad_args")
        try:
            t.encode("utf-8")
        except UnicodeEncodeError:
            raise ToolError("text is not valid unicode", code="bad_args")


_BLANKISH = frozenset("\u3164\u115f\u1160\uffa0\u2800\u17b4\u17b5")


def _display(line: str, *, crlf: bool = False) -> str:
    """Render text for the confirmation card: every invisible, blank-looking or
    control character becomes a visible \\uXXXX escape. A trailing \\r is hidden only
    when the whole file uses consistent CRLF endings (crlf=True)."""
    if crlf and line.endswith("\r"):
        line = line[:-1]
    out = []
    prev = " "
    for c in line:
        cat = unicodedata.category(c)
        bad = (cat in ("Cc", "Cf", "Zl", "Zp", "Cs", "Co") or (cat == "Zs" and c != " ")
               or c in _BLANKISH or (cat == "Mn" and (prev.isspace() or prev == "/")))
        if bad and c != "\t":
            out.append(f"\\u{ord(c):04x}" if ord(c) <= 0xFFFF else f"\\U{ord(c):08x}")
        else:
            out.append(c)
        prev = c
    return "".join(out)


def _is_crlf(text: str) -> bool:
    n = text.count("\r\n")
    return n > 0 and text.count("\r") == n and text.count("\n") == n


def _lines(text: str, crlf: bool) -> list[str]:
    return [_display(x, crlf=crlf) for x in text.split("\n")]


def plan_edit(root: Path, path: str, old_text: str, new_text: str) -> dict:
    _check_edit_text(old_text, new_text)
    real = resolve(root, path)
    if real.is_dir():
        raise ToolError("that is a folder, not a file", code="not_a_file")
    text, data = _load_text(real)
    count = text.count(old_text)
    if count != 1:
        raise ToolError(f"old_text must match exactly once; it matched {count} times", code="match_count")
    r = rel(root, real)
    new = text.replace(old_text, new_text, 1)
    if len(new.encode("utf-8")) > MAX_EDIT_BYTES:
        raise ToolError("too_large")
    if not os.stat(real).st_mode & stat.S_IWUSR:
        raise ToolError("that file is read-only", code="read_only")
    crlf = _is_crlf(text) and _is_crlf(new)
    shown = _display(r)
    diff = "\n".join(difflib.unified_diff(_lines(text, crlf), _lines(new, crlf),
                                          fromfile=shown, tofile=shown, lineterm=""))
    if old_text != new_text and diff == "":
        raise ToolError("change not representable")
    return {"kind": "edit", "path": r, "old_text": old_text, "new_text": new_text,
            "summary": f"Edit {shown}", "diff": diff, "sha256": hashlib.sha256(data).hexdigest()}


def _backup_dir(root: Path) -> Path:
    rr = _root_real(root)
    bd = rr / ".backups"
    if _lexists(bd):
        if os.path.islink(bd) or not bd.is_dir() or _blocked(rr, Path(os.path.realpath(bd))) not in (None, "hidden"):
            raise ToolError("backup folder is unusable", code="io_error")
        if Path(os.path.realpath(bd)) != bd:
            raise ToolError("backup folder is unusable", code="io_error")
    return bd


def apply_edit(root: Path, plan: dict) -> dict:
    """Re-validate and apply a plan. The atomic replace creates a new inode, so any
    other hard links to the edited file keep the old content."""
    if not isinstance(plan, dict) or plan.get("kind") != "edit" or not isinstance(plan.get("path"), str) \
            or not isinstance(plan.get("sha256"), str):
        raise ToolError("invalid edit plan", code="bad_plan")
    old_text, new_text = plan.get("old_text"), plan.get("new_text")
    _check_edit_text(old_text, new_text)
    real = _resolve_exact(root, plan["path"])
    text, data = _load_text(real)
    if hashlib.sha256(data).hexdigest() != plan["sha256"]:
        raise ToolError("the file changed since the edit was proposed; nothing was changed", code="stale")
    count = text.count(old_text)
    if count != 1:
        raise ToolError(f"old_text must match exactly once; it matched {count} times", code="match_count")
    new_bytes = text.replace(old_text, new_text, 1).encode("utf-8")
    if len(new_bytes) > MAX_EDIT_BYTES:
        raise ToolError("too_large")
    mode = stat.S_IMODE(os.stat(real).st_mode)
    if not mode & stat.S_IWUSR:
        raise ToolError("that file is read-only", code="read_only")
    r = rel(root, real)
    try:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        bd = _backup_dir(root)
        n = 0
        while True:
            folder = bd / (stamp if n == 0 else f"{stamp}-{n}")
            target = folder / r
            if not _lexists(target):
                break
            n += 1
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "xb") as bf:
            bf.write(data)
        shutil.copystat(real, target)
        fd, tmp = tempfile.mkstemp(dir=real.parent, prefix=".tmp-edit-")
        try:
            with os.fdopen(fd, "wb") as tf:
                tf.write(new_bytes)
                tf.flush()
                os.fsync(tf.fileno())
            os.chmod(tmp, mode)
            # last check: target must still be the same regular file we verified
            if not stat.S_ISREG(os.lstat(real).st_mode):
                raise ToolError("the file changed since the edit was proposed; nothing was changed", code="stale")
            os.replace(tmp, real)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    except ToolError:
        raise
    except OSError as e:
        raise ToolError(f"edit failed: {e.strerror or 'filesystem error'}", code="io_error")
    return {"backup": rel(root, target)}

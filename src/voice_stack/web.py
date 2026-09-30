"""Web search + SSRF-guarded page fetch. The only module that makes outbound network requests.

Fetch safety model:
  * `_vet` parses the URL strictly, normalises every numeric host form, resolves the host ONCE,
    and requires EVERY returned address to be public.
  * The request is then sent to the vetted IP literal (Host header + TLS SNI keep the hostname),
    so a DNS rebind between check and connect cannot reach a private address.
  * Redirects are followed manually (max 3) and each hop is fully re-vetted.
  * One overall deadline, a 2 MB cap on raw AND decoded bytes (we decode compression ourselves
    with a bounded decompressor), and a ~6000-char text cap.
"""
from __future__ import annotations

import asyncio
import codecs
import ipaddress
import logging
import time
import re
import socket
import threading
import unicodedata
import zlib
import concurrent.futures
from html.parser import HTMLParser
from typing import Callable
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx

MAX_REDIRECTS = 3
TOTAL_TIMEOUT = 10.0  # seconds, for the whole fetch (all hops, connect + headers + body)
SEARCH_TIMEOUT = 15.0
MAX_BYTES = 2 * 1024 * 1024
MAX_CHARS = 6000
MAX_URL_LEN = 2048
MAX_QUERY_LEN = 300
USER_AGENT = "Mozilla/5.0 (compatible; voice-stack-assistant/0.1)"

_DEFAULT_PORTS = {"http": 80, "https": 443}
ALLOWED_PORTS = frozenset({80, 443, 8080, 8443})  # deliberate: no SSH/SMTP/DB/etc. ports even on public IPs
MAX_EXTRACT_CHARS = 128 * 1024  # HTML fed to trafilatura (its cost is ~quadratic in element count)
MAX_EXTRACT_TAGS = 8000  # more '<' than this in the window -> skip trafilatura, use the linear strip
EXTRACTOR_WAIT = 2.0  # seconds a fetch may wait for the single extraction slot before failing fast
TITLE_SCAN_CHARS = 64 * 1024
_NUMERIC_LABEL = re.compile(r"^(?:0x[0-9a-f]*|\d+)$", re.I)
_STRICT_NUMERIC_LABEL = re.compile(r"^(?:0x[0-9a-f]+|\d+)$", re.I)
_DNS_LABEL = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_BLOCKED_SUFFIXES = (".localhost", ".local", ".internal", ".localdomain", ".home.arpa", ".in-addr.arpa", ".ip6.arpa")

_NAT64 = ipaddress.ip_network("64:ff9b::/96")
_NAT64_LOCAL = ipaddress.ip_network("64:ff9b:1::/48")
_6TO4 = ipaddress.ip_network("2002::/16")
_TEREDO = ipaddress.ip_network("2001::/32")
_V4_COMPAT = ipaddress.ip_network("::/96")


_log = logging.getLogger(__name__)
_GLOBAL_V6 = ipaddress.ip_network("2000::/3")
_LOCAL_TTL = 30.0
_local_cache: tuple[float, list] = (0.0, [])


def _local_addresses() -> list:
    """This machine's own interface addresses (cached ~30 s).

    Tests replace this function wholesale (monkeypatch `web._local_addresses`); it is deliberately
    NOT a production parameter. Enumeration failure is logged and treated as 'none known'.
    """
    global _local_cache
    now = time.monotonic()
    if _local_cache[1] and now - _local_cache[0] < _LOCAL_TTL:
        return _local_cache[1]
    out: list = []
    try:
        import psutil
        for addrs in psutil.net_if_addrs().values():
            for a in addrs:
                if a.family in (socket.AF_INET, socket.AF_INET6):
                    try:
                        out.append(ipaddress.ip_address(a.address.split("%", 1)[0]))
                    except ValueError:
                        pass
    except Exception:
        _log.warning("could not enumerate local interface addresses", exc_info=True)
    _local_cache = (now, out)
    return out


def _is_local(ip) -> bool:
    """True if `ip` is one of our own addresses, or shares a /64 with one of our global IPv6 addresses."""
    try:
        local = _local_addresses()
    except Exception:
        _log.warning("local address provider failed", exc_info=True)
        return False
    for la in local:
        if ip == la:
            return True
        if ip.version == 6 and la.version == 6 and la in _GLOBAL_V6 and (int(ip) >> 64) == (int(la) >> 64):
            return True
    return False


class WebError(Exception):
    """Model-readable failure message."""


# --------------------------------------------------------------------------- SSRF guard

def _is_public(ip) -> bool:
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            return _is_public(ip.ipv4_mapped)
        if ip in _NAT64:
            return _is_public(ipaddress.IPv4Address(ip.packed[-4:]))
        if ip in _6TO4:
            return _is_public(ipaddress.IPv4Address(ip.packed[2:6]))
        if ip in _NAT64_LOCAL or ip in _TEREDO or ip in _V4_COMPAT:
            return False
        if ip not in _GLOBAL_V6:  # fec0::/10 site-local, fc00::/7, fe80::/10, ... only 2000::/3 is public unicast
            return False
    if _is_local(ip):
        return False
    return bool(
        ip.is_global
        and not (ip.is_multicast or ip.is_loopback or ip.is_private or ip.is_link_local
                 or ip.is_reserved or ip.is_unspecified)
    )


class _Target:
    __slots__ = ("url", "scheme", "host", "port", "path_query", "ips", "is_literal")


def _bad(reason: str) -> WebError:
    return WebError(f"blocked_url: {reason}")


def _parse(url) -> tuple[str, str, int, str, object | None]:
    """Strictly parse `url`. Returns (scheme, host, port, path_query, literal_ip_or_None)."""
    if not isinstance(url, str) or not url:
        raise _bad("empty or non-string URL")
    if len(url) > MAX_URL_LEN:
        raise _bad("URL too long")
    for ch in url:
        if ch == "\\" or unicodedata.category(ch)[0] in "CZ":
            raise _bad("whitespace, control or backslash character in URL")
    try:
        parts = urlsplit(url)
    except ValueError:
        raise _bad("unparseable URL")
    scheme = parts.scheme.lower()
    if scheme not in _DEFAULT_PORTS:
        raise _bad("only http and https are allowed")
    netloc = parts.netloc
    if not netloc:
        raise _bad("missing host")
    if "@" in netloc:
        raise _bad("credentials in URL are not allowed")
    try:
        port = parts.port
    except ValueError:
        raise _bad("invalid port")
    if port is None:
        port = _DEFAULT_PORTS[scheme]
    if port not in ALLOWED_PORTS:
        raise _bad("port not allowed (only 80, 443, 8080, 8443)")
    try:
        host = parts.hostname
    except ValueError:
        raise _bad("invalid host")
    if not host:
        raise _bad("missing host")
    if "%" in host:
        raise _bad("percent-escapes or zone ids in host are not allowed")

    literal = None
    if ":" in host:  # IPv6 literal (brackets already stripped by urlsplit)
        if not netloc.startswith("["):
            raise _bad("invalid host")
        try:
            literal = ipaddress.IPv6Address(host)
        except ValueError:
            raise _bad("invalid IPv6 literal")
        host = str(literal)
    else:
        if netloc.startswith("[") or "[" in netloc or "]" in netloc:
            raise _bad("invalid host")
        try:
            host = host.encode("idna").decode("ascii").lower()  # NFKC + punycode; also folds full-width/ideographic dots
        except (UnicodeError, ValueError):
            raise _bad("invalid host name")
        if host.endswith("."):
            host = host[:-1]
        if not host or host.endswith("."):
            raise _bad("invalid host name")
        labels = host.split(".")
        if _NUMERIC_LABEL.match(labels[-1]):
            # looks like an IPv4 form (decimal / hex / octal / short): parse like inet_aton, else refuse
            if len(labels) > 4 or not all(_STRICT_NUMERIC_LABEL.match(x) for x in labels):
                raise _bad("malformed numeric host")
            try:
                literal = ipaddress.IPv4Address(socket.inet_aton(host))
            except (OSError, ValueError):
                raise _bad("malformed numeric host")
            host = str(literal)
        else:
            if host == "localhost" or host.endswith(_BLOCKED_SUFFIXES) or len(labels) < 2:
                raise _bad("local host names are not allowed")
            if not all(_DNS_LABEL.match(x) for x in labels):
                raise _bad("invalid host name")
    path_query = parts.path or "/"
    if parts.query:
        path_query += "?" + parts.query
    return scheme, host, port, path_query, literal


def _resolve(host: str, port: int, resolver: Callable) -> list:
    try:
        infos = resolver(host, port, type=socket.SOCK_STREAM)
    except Exception:
        raise _bad("host did not resolve")
    if not infos:
        raise _bad("host did not resolve")
    ips = []
    for info in infos:
        try:
            raw = info[4][0]
            if not isinstance(raw, str) or "%" in raw:
                raise ValueError
            ips.append(ipaddress.ip_address(raw))
        except (TypeError, IndexError, ValueError):
            raise _bad("host resolved to an unusable address")
    return ips


def _vet(url, resolver: Callable) -> _Target:
    scheme, host, port, path_query, literal = _parse(url)
    ips = [literal] if literal is not None else _resolve(host, port, resolver)
    for ip in ips:
        if not _is_public(ip):
            raise _bad("host is not a public internet address")  # ONE bad address rejects the whole host
    t = _Target()
    t.url, t.scheme, t.host, t.port, t.path_query, t.ips, t.is_literal = url, scheme, host, port, path_query, ips, literal is not None
    return t


def check_url(url: str, resolver: Callable = socket.getaddrinfo) -> str:
    """Return `url` unchanged if it is http(s) and every address it resolves to is public; else WebError."""
    _vet(url, resolver)
    return url


# --------------------------------------------------------------------------- fetch

def _pinned_request(t: _Target, client: httpx.AsyncClient) -> httpx.Request:
    ip = sorted(t.ips, key=lambda a: a.version != 4)[0]  # prefer IPv4, connect ONLY to this vetted address
    ip_host = f"[{ip}]" if ip.version == 6 else str(ip)
    default = _DEFAULT_PORTS[t.scheme]
    netloc = ip_host if t.port == default else f"{ip_host}:{t.port}"
    path, _, query = t.path_query.partition("?")
    pinned = urlunsplit((t.scheme, netloc, path, query, ""))
    host_header = f"[{t.host}]" if ":" in t.host else t.host
    if t.port != default:
        host_header += f":{t.port}"
    ext = {"sni_hostname": t.host} if t.scheme == "https" and not t.is_literal else {}
    return client.build_request(
        "GET", pinned,
        headers={"Host": host_header, "User-Agent": USER_AGENT, "Accept": "text/html,text/plain;q=0.9",
                 "Accept-Encoding": "gzip, deflate"},
        extensions=ext,
    )


def _decoder(encoding: str):
    enc = (encoding or "").strip().lower()
    if enc in ("", "identity"):
        return None
    if enc in ("gzip", "x-gzip"):
        return zlib.decompressobj(zlib.MAX_WBITS | 16)
    if enc == "deflate":
        return zlib.decompressobj()
    raise WebError("fetch_failed: unsupported content-encoding")


async def _read_body(resp: httpx.Response) -> tuple[bytes, bool]:
    """Read at most MAX_BYTES of DECODED and MAX_BYTES of RAW data. Returns (body, truncated)."""
    out = bytearray()
    if resp.is_stream_consumed:  # already materialised by the transport (only happens with injected test transports)
        data = resp.content
        return bytes(data[:MAX_BYTES]), len(data) > MAX_BYTES
    dec = _decoder(resp.headers.get("content-encoding", ""))
    raw_total = 0
    truncated = False
    try:
        async for raw in resp.aiter_raw():
            raw_total += len(raw)
            if raw_total > MAX_BYTES:
                raw = raw[: len(raw) - (raw_total - MAX_BYTES)]
                truncated = True
            if dec is None:
                room = MAX_BYTES - len(out)
                out += raw[:room]
                if len(raw) > room:
                    truncated = True
            else:
                data = raw
                while data and not dec.eof:
                    out += dec.decompress(data, MAX_BYTES - len(out) + 1)  # bounded output per call
                    data = dec.unconsumed_tail
                    if len(out) > MAX_BYTES:
                        del out[MAX_BYTES:]
                        truncated = True
                        break
                if dec.eof and dec.unused_data:
                    break
            if truncated or len(out) >= MAX_BYTES:
                truncated = truncated or len(out) >= MAX_BYTES
                break
    except zlib.error:
        raise WebError("fetch_failed: corrupt compressed body")
    return bytes(out), truncated


def _charset(content_type: str) -> str:
    m = re.search(r"charset\s*=\s*[\"']?([\w.:-]+)", content_type, re.I)
    name = m.group(1) if m else "utf-8"
    try:
        info = codecs.lookup(name)
        if not getattr(info, "_is_text_encoding", True):  # idna, rot13, base64, zlib, ...
            return "utf-8"
        return info.name
    except (LookupError, ValueError, TypeError):
        return "utf-8"


def _decode(body: bytes, content_type: str) -> str:
    try:
        return body.decode(_charset(content_type), errors="replace")
    except (LookupError, UnicodeError, ValueError, TypeError):
        return body.decode("utf-8", errors="replace")


_TITLE_RE = re.compile(r"<title[^>]{0,256}>([^<]{0,1000})", re.I)


def _title(html: str) -> str:
    import html as _html
    m = _TITLE_RE.search(html[:TITLE_SCAN_CHARS])
    return _clean(_html.unescape(m.group(1))) if m else ""


def _clean(s: str) -> str:
    """Drop control / format (bidi, zero-width, tag) chars; collapse whitespace."""
    s = "".join(c if c in "\n" or unicodedata.category(c) not in ("Cc", "Cf") else " " for c in s)
    return " ".join(s.split())


def _extract(html: str) -> tuple[str, str]:
    import trafilatura
    html = html[:MAX_EXTRACT_CHARS]
    text = trafilatura.extract(html, include_comments=False) or ""
    title = _title(html)
    if not title:
        try:
            meta = trafilatura.extract_metadata(html)
            title = _clean((meta.title or "") if meta is not None else "")
        except Exception:
            title = ""
    return text, title


# ---- extraction slot: at most ONE extraction thread alive in the whole process.
# A plain daemon thread per job (not a ThreadPoolExecutor: its workers are joined at interpreter exit,
# so a stuck extraction would block shutdown). A threading.Lock (not per-loop) is released only when the
# thread really finishes, so a timed-out extraction keeps the slot busy and later fetches fail fast with
# 'extractor_busy' instead of stacking threads. DNS vetting and web_search use the default executor.
_extract_slot = threading.Lock()


class _Strip(HTMLParser):
    _SKIP = {"script", "style", "noscript", "template", "svg", "head"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in self._SKIP and self.skip:
            self.skip -= 1

    def handle_data(self, data):
        if not self.skip:
            self.out.append(data)


def _strip_extract(html: str) -> tuple[str, str]:
    """Linear-time fallback for hostile markup: plain text of the body, no trafilatura."""
    html = html[:MAX_EXTRACT_CHARS]
    p = _Strip()
    try:
        p.feed(html)
        p.close()
    except Exception:
        pass
    return " ".join(" ".join(p.out).split()), _title(html)


def _is_hostile(html: str) -> bool:
    return html[:MAX_EXTRACT_CHARS].count("<") > MAX_EXTRACT_TAGS


async def _run_extraction(fn: Callable, html: str):
    """Run fn(html) on the dedicated extraction thread; fail fast if the slot is taken."""
    deadline = time.monotonic() + EXTRACTOR_WAIT
    while not _extract_slot.acquire(blocking=False):
        if time.monotonic() >= deadline:
            raise WebError("extractor_busy")
        await asyncio.sleep(0.05)
    cf: concurrent.futures.Future = concurrent.futures.Future()

    def work():
        try:
            try:
                res = fn(html)
            except BaseException as e:  # noqa: BLE001
                if not cf.cancelled():
                    cf.set_exception(e)
            else:
                if not cf.cancelled():
                    cf.set_result(res)
        except concurrent.futures.InvalidStateError:
            pass  # the awaiting fetch timed out and cancelled the future; result is discarded
        finally:
            _extract_slot.release()  # only when the thread has truly finished

    try:
        threading.Thread(target=work, name="extract", daemon=True).start()
    except BaseException:
        _extract_slot.release()
        raise
    return await asyncio.wrap_future(cf)


async def _fetch_chain(url: str, client: httpx.AsyncClient, resolver: Callable) -> tuple[str, str, bytes, bool]:
    current = url
    for hop in range(MAX_REDIRECTS + 1):
        t = await asyncio.to_thread(_vet, current, resolver)  # resolver may block
        req = _pinned_request(t, client)
        resp = await client.send(req, stream=True)
        try:
            if resp.status_code in (301, 302, 303, 307, 308):
                loc = resp.headers.get("location")
                if not loc:
                    raise WebError("fetch_failed: redirect without Location")
                if hop == MAX_REDIRECTS:
                    raise WebError("fetch_failed: too many redirects")
                current = urljoin(current, loc)
                continue
            if resp.status_code >= 400:
                raise WebError(f"fetch_failed: HTTP {resp.status_code}")
            if resp.status_code != 200:
                raise WebError(f"fetch_failed: unexpected HTTP {resp.status_code}")
            ctype = resp.headers.get("content-type", "")
            media = ctype.split(";", 1)[0].strip().lower()
            if media not in ("text/html", "text/plain"):
                raise WebError(f"fetch_failed: unsupported content type '{re.sub(r'[^A-Za-z0-9/+.-]', '', media)[:60] or 'none'}'")
            body, truncated = await _read_body(resp)
            return current, ctype, body, truncated
        finally:
            await resp.aclose()
    raise WebError("fetch_failed: too many redirects")  # unreachable


async def fetch_page(url: str, *, client: httpx.AsyncClient | None = None,
                     resolver: Callable = socket.getaddrinfo, max_chars: int = MAX_CHARS) -> dict:
    """Fetch one page as plain text. Keys: url, title, text, truncated (+ note when no text).

    Only raises WebError. `client` is for tests with a MockTransport only (production passes none);
    a client that follows redirects or reads proxy env vars would bypass per-hop vetting / pinning, so it is refused.
    """
    try:
        max_chars = max(1, min(int(max_chars), MAX_CHARS))
    except (TypeError, ValueError, OverflowError):
        raise WebError("invalid_request: max_chars must be an integer")
    own = client is None
    if not own and (client.follow_redirects or client.trust_env):
        raise WebError("invalid_client: injected client must have follow_redirects=False and trust_env=False")
    if own:
        # trust_env=False: never route through an env proxy (that would defeat IP pinning)
        client = httpx.AsyncClient(timeout=httpx.Timeout(TOTAL_TIMEOUT), follow_redirects=False, trust_env=False,
                                   limits=httpx.Limits(max_connections=2, max_keepalive_connections=0))
    try:
        try:
            async with asyncio.timeout(TOTAL_TIMEOUT):  # one deadline: network AND extraction
                final_url, ctype, body, truncated = await _fetch_chain(url, client, resolver)
                decoded = _decode(body, ctype).replace("\x00", "")
                media = ctype.split(";", 1)[0].strip().lower()
                hostile = False
                if media == "text/plain":
                    text, title = decoded.strip(), ""
                else:
                    try:
                        hostile = _is_hostile(decoded)
                        text, title = await _run_extraction(_strip_extract if hostile else _extract, decoded)
                    except (asyncio.CancelledError, WebError):
                        raise
                    except Exception:
                        text, title = "", ""
        except TimeoutError:
            raise WebError("fetch_failed: timeout")
        except WebError:
            raise
        except Exception as e:  # never leak a raw exception type to callers
            raise WebError(f"fetch_failed: {re.sub(r'[^A-Za-z0-9_]', '', type(e).__name__)}")
    finally:
        if own:
            await client.aclose()
    page = {"url": final_url, "title": title[:300], "text": text, "truncated": truncated}
    if len(text) > max_chars:
        page["text"], page["truncated"] = text[:max_chars], True
    if hostile:
        page["note"] = "page markup was unusually dense; plain-text fallback used"
    elif not page["text"]:
        page["note"] = "no readable text could be extracted from this page"
    return page


# --------------------------------------------------------------------------- search

def _ddgs_backend(query: str, n: int) -> list:
    from ddgs import DDGS
    return DDGS().text(query, max_results=n)


async def web_search(query: str, n: int = 5, backend: Callable | None = None) -> list[dict]:
    """Search the web. Returns [{title, url, snippet}]. Any backend failure -> WebError('search_unavailable')."""
    if not isinstance(query, str):
        raise WebError("invalid_query: query must be text")
    query = query.strip()
    if not query or len(query) > MAX_QUERY_LEN:
        raise WebError(f"invalid_query: query must be 1 to {MAX_QUERY_LEN} characters")
    if any(unicodedata.category(c) == "Cc" for c in query):
        raise WebError("invalid_query: control characters are not allowed")
    if isinstance(n, bool) or not isinstance(n, int):
        raise WebError("invalid_query: n must be an integer")
    n = max(1, min(n, 10))
    fn = backend or _ddgs_backend
    try:
        raw = await asyncio.wait_for(asyncio.to_thread(fn, query, n), SEARCH_TIMEOUT)
        if not isinstance(raw, list):
            raise TypeError("backend returned non-list")
        results = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            link = item.get("href") or item.get("url")
            if not isinstance(link, str) or not link.lower().startswith(("http://", "https://")):
                continue
            results.append({
                "title": _clean(str(item.get("title") or ""))[:300],
                "url": link[:MAX_URL_LEN],
                "snippet": _clean(str(item.get("body") or item.get("snippet") or ""))[:600],
            })
            if len(results) >= n:
                break
        return results
    except Exception:
        raise WebError("search_unavailable")


# --------------------------------------------------------------------------- untrusted wrapper

def wrap_untrusted(text: str) -> str:
    """Wrap web text so the model treats it as data.

    NFKC-normalise (folds full-width look-alikes), strip all format chars (bidi/zero-width/soft-hyphen)
    and Unicode tag characters U+E0000-E007F (hidden instructions), then escape EVERY '<' and '>' so no
    tag (real or look-alike, open or close) can exist inside the body.
    """
    text = unicodedata.normalize("NFKC", str(text))
    text = "".join(c for c in text
                   if unicodedata.category(c) != "Cf" and not 0xE0000 <= ord(c) <= 0xE007F)
    text = text.replace("<", "&lt;").replace(">", "&gt;")
    return "<untrusted_web_content>\n" + text + "\n</untrusted_web_content>"

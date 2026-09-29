"""
Manage `mlx_lm.server` as a subprocess, serving the LLM over a local HTTP
API that OpenAILLMService talks to (see PLAN.md: "LLM out of process").

The subprocess runs in its own session (start_new_session=True) so it is not
in our process group; we are responsible for terminating it ourselves, in a
`finally` block covering normal exit, Ctrl-C, and crashes -- an orphaned
mlx_lm.server holds ~20GB of GPU memory.
"""

import subprocess
import sys
import time
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_PATH = REPO_ROOT / "mlx_lm_server.log"


class MLXLMServer:
    """Starts, polls readiness for, and stops an `mlx_lm.server` subprocess."""

    def __init__(self, model_id: str, host: str = "127.0.0.1", port: int = 8080):
        self._model_id = model_id
        self._host = host
        self._port = port
        self._process: subprocess.Popen | None = None
        self._log_file = None

    @property
    def base_url(self) -> str:
        return f"http://{self._host}:{self._port}/v1"

    def start(self, timeout: float = 60.0) -> None:
        self._close_log()
        log_file = open(LOG_PATH, "w")
        self._log_file = log_file
        self._process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "mlx_lm",
                "server",
                "--model",
                self._model_id,
                "--host",
                self._host,
                "--port",
                str(self._port),
            ],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self._wait_ready(timeout)

    def _wait_ready(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        url = f"{self.base_url}/models"
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                raise RuntimeError(
                    f"mlx_lm.server exited early (code={self._process.returncode}); "
                    f"see {LOG_PATH}. A stale mlx_lm.server may already be holding "
                    f"port {self._port} -- try `pkill -f mlx_lm.server` and retry."
                )
            try:
                resp = httpx.get(url, timeout=2.0)
                if resp.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
        self.stop()
        raise TimeoutError(
            f"mlx_lm.server did not become ready within {timeout}s; see {LOG_PATH}. "
            f"A stale mlx_lm.server may already be holding port {self._port} -- try "
            f"`pkill -f mlx_lm.server` and retry."
        )

    def _close_log(self) -> None:
        if self._log_file is not None:
            try:
                self._log_file.close()
            finally:
                self._log_file = None

    def stop(self, grace_period: float = 5.0) -> None:
        """Terminate the subprocess, escalating to kill after a grace period."""
        try:
            if self._process is None:
                return
            if self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=grace_period)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=grace_period)
            self._process = None
        finally:
            self._close_log()

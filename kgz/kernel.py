"""
Core Kernel class — persistent WebSocket connection to a Kaggle Jupyter kernel.

Features:
- Persistent connection (reuses WebSocket across calls)
- Auto-reconnect on timeout/disconnect
- Structured output (CellResult with stdout, stderr, return_value, errors)
- Streaming with event callbacks
- Environment variable injection
- Session save/resume
"""

import json
import uuid
import time
import os
import urllib.request
from dataclasses import dataclass, field
from urllib.parse import urlparse

import websocket


# ---------------------------------------------------------------------------
# Structured output
# ---------------------------------------------------------------------------
@dataclass
class CellResult:
    """Structured result from executing a cell."""
    success: bool = True
    stdout: str = ""
    stderr: str = ""
    return_value: str = None
    error_name: str = None
    error_value: str = None
    traceback: list = field(default_factory=list)
    display_data: list = field(default_factory=list)  # images, HTML, etc.
    execution_count: int = None
    elapsed_seconds: float = 0.0

    @property
    def output(self) -> str:
        """Combined stdout + return_value."""
        parts = []
        if self.stdout:
            parts.append(self.stdout)
        if self.return_value:
            parts.append(self.return_value)
        return "".join(parts)

    def __repr__(self):
        status = "OK" if self.success else f"ERROR: {self.error_name}"
        out = self.output[:60] + "..." if len(self.output) > 60 else self.output
        return f"CellResult({status}, {out!r})"


class KernelError(Exception):
    """Raised when remote code execution fails."""
    def __init__(self, result: CellResult):
        self.result = result
        super().__init__(f"{result.error_name}: {result.error_value}")


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------
class Kernel:
    """
    Persistent connection to a Kaggle Jupyter kernel.

    Features:
    - Reuses WebSocket across execute() calls (no reconnect overhead)
    - Auto-reconnects on disconnect/timeout
    - Returns structured CellResult with parsed output
    - Supports streaming with callbacks
    - Session save/resume

    Usage:
        k = Kernel("https://kkb-production.jupyter-proxy.kaggle.net/k/.../proxy")
        result = k.execute("print('hello')")
        print(result.stdout)    # "hello\n"
        print(result.success)   # True
    """

    def __init__(self, url: str, name: str = None):
        """
        Connect to a Kaggle Jupyter kernel.

        Args:
            url: Kaggle Jupyter proxy URL ending in /proxy
            name: Optional session name for save/resume
        """
        self.base_url = url.rstrip("/")
        parsed = urlparse(self.base_url)
        self._ws_host = parsed.netloc + parsed.path
        self._http_url = f"https://{parsed.netloc}{parsed.path}"
        self.name = name or f"kgz-{int(time.time())}"
        self.kernel_id = self._discover_kernel()
        self._ws: object = None
        self._history: list = []
        self._max_retries = 3

    def _discover_kernel(self) -> str:
        """Find the running kernel via REST API."""
        resp = urllib.request.urlopen(f"{self._http_url}/api/kernels")
        kernels = json.loads(resp.read())
        if not kernels:
            raise ConnectionError("No running kernels found")
        return kernels[0]["id"]

    @property
    def _ws_url(self) -> str:
        return f"wss://{self._ws_host}/api/kernels/{self.kernel_id}/channels"

    # ----------------------------------------------------------------
    # Persistent WebSocket connection
    # ----------------------------------------------------------------
    def _connect(self):
        """Create or reconnect WebSocket."""
        if self._ws is not None:
            try:
                self._ws.ping()
                return  # Still alive
            except Exception:
                self._close_ws()

        self._ws = websocket.create_connection(self._ws_url, timeout=30)

    def _close_ws(self):
        if getattr(self, '_ws', None):
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None

    def close(self):
        """Close the persistent connection."""
        self._close_ws()

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ----------------------------------------------------------------
    # Core execute with structured output
    # ----------------------------------------------------------------
    def execute(
        self,
        code: str,
        timeout: int = 600,
        stream: bool = True,
        raise_on_error: bool = False,
        on_event: object = None,
    ) -> CellResult:
        """
        Execute Python code on the remote kernel.

        Args:
            code: Python code string
            timeout: Max seconds to wait
            stream: Print output in real-time
            raise_on_error: Raise KernelError on remote exceptions
            on_event: Callback(event_type: str, data: dict) for each message

        Returns:
            CellResult with structured output
        """
        for attempt in range(self._max_retries):
            try:
                return self._execute_impl(code, timeout, stream, raise_on_error, on_event)
            except (websocket.WebSocketConnectionClosedException, ConnectionError, OSError) as e:
                self._close_ws()
                if attempt < self._max_retries - 1:
                    time.sleep(1)
                    continue
                raise ConnectionError(f"Failed after {self._max_retries} retries: {e}")

    def _execute_impl(self, code, timeout, stream, raise_on_error, on_event) -> CellResult:
        self._connect()
        msg_id = str(uuid.uuid4())
        t0 = time.time()

        self._ws.send(json.dumps({
            "header": {
                "msg_id": msg_id,
                "msg_type": "execute_request",
                "username": "",
                "session": str(uuid.uuid4()),
                "version": "5.3",
            },
            "parent_header": {},
            "metadata": {},
            "content": {
                "code": code,
                "silent": False,
                "store_history": True,
                "user_expressions": {},
                "allow_stdin": False,
                "stop_on_error": True,
            },
            "channel": "shell",
        }))

        result = CellResult()
        stdout_parts = []
        stderr_parts = []

        while time.time() - t0 < timeout:
            try:
                self._ws.settimeout(min(10, timeout - (time.time() - t0)))
                raw = self._ws.recv()
                msg = json.loads(raw)
                msg_type = msg.get("msg_type", "")
                parent_id = msg.get("parent_header", {}).get("msg_id", "")

                if parent_id != msg_id:
                    continue

                if on_event:
                    on_event(msg_type, msg.get("content", {}))

                if msg_type == "stream":
                    text = msg["content"]["text"]
                    name = msg["content"].get("name", "stdout")
                    if name == "stderr":
                        stderr_parts.append(text)
                    else:
                        stdout_parts.append(text)
                    if stream:
                        print(text, end="", flush=True)

                elif msg_type == "execute_result":
                    result.return_value = msg["content"]["data"].get("text/plain", "")
                    result.execution_count = msg["content"].get("execution_count")
                    if stream and result.return_value:
                        print(result.return_value, end="", flush=True)

                elif msg_type == "display_data":
                    data = msg["content"]["data"]
                    result.display_data.append(data)
                    text = data.get("text/plain", "")
                    if stream and text:
                        print(text, end="", flush=True)

                elif msg_type == "error":
                    result.success = False
                    result.error_name = msg["content"]["ename"]
                    result.error_value = msg["content"]["evalue"]
                    result.traceback = msg["content"].get("traceback", [])
                    if stream:
                        print(f"\nERROR: {result.error_name}: {result.error_value}")
                    break

                elif msg_type == "execute_reply":
                    status = msg["content"].get("status", "ok")
                    if status == "error":
                        result.success = False
                        result.error_name = msg["content"].get("ename", "")
                        result.error_value = msg["content"].get("evalue", "")
                    result.execution_count = msg["content"].get("execution_count")
                    break

            except websocket.WebSocketTimeoutException:
                continue

        result.stdout = "".join(stdout_parts)
        result.stderr = "".join(stderr_parts)
        result.elapsed_seconds = time.time() - t0

        # Save to history
        self._history.append({
            "code": code,
            "success": result.success,
            "output": result.output[:200],
            "elapsed": result.elapsed_seconds,
        })

        if raise_on_error and not result.success:
            raise KernelError(result)

        return result

    # ----------------------------------------------------------------
    # Status / control
    # ----------------------------------------------------------------
    def status(self) -> str:
        """Get kernel state: 'idle' or 'busy'."""
        resp = urllib.request.urlopen(f"{self._http_url}/api/kernels/{self.kernel_id}")
        return json.loads(resp.read())["execution_state"]

    def interrupt(self):
        """Interrupt running execution."""
        req = urllib.request.Request(
            f"{self._http_url}/api/kernels/{self.kernel_id}/interrupt",
            method="POST",
        )
        urllib.request.urlopen(req)

    def wait(self, poll_interval: int = 15, timeout: int = 3600):
        """Block until kernel is idle."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.status() == "idle":
                return
            time.sleep(poll_interval)
        raise TimeoutError(f"Kernel still busy after {timeout}s")

    def is_alive(self) -> bool:
        """Check if kernel is reachable."""
        try:
            self.status()
            return True
        except Exception:
            return False

    def restart(self):
        """Restart the kernel (clears all state)."""
        req = urllib.request.Request(
            f"{self._http_url}/api/kernels/{self.kernel_id}/restart",
            method="POST",
        )
        urllib.request.urlopen(req)
        self._close_ws()
        time.sleep(2)

    # ----------------------------------------------------------------
    # Environment
    # ----------------------------------------------------------------
    def set_env(self, **env_vars):
        """
        Set environment variables on the remote kernel.
        Secrets are NOT saved to execution history or notebook export.
        """
        code = "\n".join(f"import os; os.environ[{k!r}] = {v!r}" for k, v in env_vars.items())
        result = self.execute(code, stream=False)
        # Remove from history — secrets should never be exported
        if self._history and self._history[-1].get("code", "").startswith("import os; os.environ"):
            self._history.pop()
        return result

    # ----------------------------------------------------------------
    # Snapshot / inspection
    # ----------------------------------------------------------------
    def snapshot(self) -> dict:
        """
        Get a snapshot of all variables in the remote kernel.
        Returns dict of {name: {"type": ..., "shape": ..., "repr": ...}}.
        """
        result = self.execute("""
import json
_kgz_snap = {}
for _n, _v in list(globals().items()):
    if _n.startswith('_') or _n in ('In', 'Out', 'get_ipython', 'exit', 'quit'):
        continue
    info = {"type": type(_v).__name__}
    if hasattr(_v, 'shape'):
        info["shape"] = str(_v.shape)
    if hasattr(_v, '__len__') and not isinstance(_v, str):
        try: info["len"] = len(_v)
        except: pass
    try: info["repr"] = repr(_v)[:100]
    except: info["repr"] = "..."
    _kgz_snap[_n] = info
print(json.dumps(_kgz_snap))
del _kgz_snap
""", stream=False)
        try:
            import ast
            return json.loads(result.stdout.strip())
        except Exception:
            return {"_raw": result.stdout}

    def resources(self) -> dict:
        """Get GPU/TPU resource usage on the remote kernel."""
        result = self.execute("""
import json
info = {}
try:
    import jax
    info["backend"] = jax.default_backend()
    info["device_count"] = jax.device_count()
    info["devices"] = str(jax.devices())
except: pass
try:
    import subprocess
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                                    "--format=csv,nounits,noheader"], text=True)
    gpus = []
    for line in out.strip().split("\\n"):
        util, used, total = line.split(", ")
        gpus.append({"utilization": int(util), "memory_used_mb": int(used), "memory_total_mb": int(total)})
    info["gpus"] = gpus
except: pass
try:
    import psutil
    info["cpu_percent"] = psutil.cpu_percent()
    mem = psutil.virtual_memory()
    info["ram_used_gb"] = round(mem.used / 1e9, 1)
    info["ram_total_gb"] = round(mem.total / 1e9, 1)
except: pass
print(json.dumps(info))
""", stream=False)
        try:
            return json.loads(result.stdout.strip())
        except Exception:
            return {"_raw": result.stdout}

    # ----------------------------------------------------------------
    # Multi-cell pipeline
    # ----------------------------------------------------------------
    def execute_notebook(
        self,
        cells: list,
        stop_on_error: bool = True,
        stream: bool = True,
    ) -> list:
        """
        Execute multiple cells sequentially, like running a notebook.

        Args:
            cells: List of code strings
            stop_on_error: Stop execution if any cell fails
            stream: Print output in real-time

        Returns:
            List of CellResult, one per cell
        """
        results = []
        for i, code in enumerate(cells):
            if stream:
                print(f"\n--- Cell {i + 1}/{len(cells)} ---")
            result = self.execute(code, stream=stream)
            results.append(result)
            if stop_on_error and not result.success:
                if stream:
                    print(f"\nStopped at cell {i + 1}: {result.error_name}")
                break
        return results

    # ----------------------------------------------------------------
    # Session persistence
    # ----------------------------------------------------------------
    def save_session(self, path: str = None):
        """Save session info for later resume."""
        if path is None:
            os.makedirs(os.path.expanduser("~/.kgz"), exist_ok=True)
            path = os.path.expanduser(f"~/.kgz/{self.name}.json")
        data = {
            "url": self.base_url,
            "kernel_id": self.kernel_id,
            "name": self.name,
            "saved_at": time.time(),
            "history_len": len(self._history),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path

    @classmethod
    def resume(cls, name: str) -> "Kernel":
        """Resume a previously saved session."""
        path = os.path.expanduser(f"~/.kgz/{name}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"No saved session: {path}")
        with open(path) as f:
            data = json.load(f)
        k = cls(data["url"], name=data["name"])
        return k

    @staticmethod
    def list_sessions() -> list:
        """List all saved sessions."""
        session_dir = os.path.expanduser("~/.kgz")
        if not os.path.exists(session_dir):
            return []
        sessions = []
        for f in os.listdir(session_dir):
            if f.endswith(".json"):
                try:
                    with open(os.path.join(session_dir, f)) as fh:
                        data = json.load(fh)
                    if "name" in data and "url" in data:
                        sessions.append(data)
                except Exception:
                    pass
        return sessions

    # ----------------------------------------------------------------
    # Notebook export
    # ----------------------------------------------------------------
    def to_notebook(self, path: str = "output.ipynb"):
        """Export execution history as a Jupyter notebook."""
        cells = []
        for entry in self._history:
            cell = {
                "cell_type": "code",
                "source": entry["code"],
                "metadata": {},
                "outputs": [],
                "execution_count": None,
            }
            if entry.get("output"):
                cell["outputs"].append({
                    "output_type": "stream",
                    "name": "stdout",
                    "text": entry["output"],
                })
            cells.append(cell)

        notebook = {
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python", "version": "3.12"},
                "kgz": {"session": self.name, "kernel_id": self.kernel_id},
            },
            "cells": cells,
        }
        with open(path, "w") as f:
            json.dump(notebook, f, indent=2)
        return path

    # ----------------------------------------------------------------
    # Cached execution
    # ----------------------------------------------------------------
    def execute_cached(self, code, **kwargs):
        """Execute with local caching. Identical code returns cached result."""
        from kgz.cache import ResultCache
        if not hasattr(self, '_cache'):
            self._cache = ResultCache()
        cached = self._cache.get(code)
        if cached is not None:
            return cached
        result = self.execute(code, **kwargs)
        if result.success:
            self._cache.put(code, result)
        return result

    def clear_cache(self):
        """Clear local result cache."""
        from kgz.cache import ResultCache
        ResultCache().clear()

    # ----------------------------------------------------------------
    # Quota tracking
    # ----------------------------------------------------------------
    def quota(self, device_type=None):
        """Get quota tracker. Auto-detects GPU/TPU."""
        from kgz.quota import QuotaTracker
        if not hasattr(self, '_quota'):
            self._quota = QuotaTracker()
        if device_type is None:
            r = self.execute("import jax; print(jax.default_backend())", stream=False)
            device_type = "tpu" if "tpu" in r.stdout.strip().lower() else "gpu"
        return self._quota, device_type

    def quota_summary(self):
        """Print quota summary."""
        qt, dt = self.quota()
        print(qt.summary(dt))
        print(f"Session limit: {qt.session_time_left(dt):.1f}h remaining")
        return qt.remaining(dt)

    def start_quota_tracking(self, device_type=None):
        """Start tracking quota for this session."""
        qt, dt = self.quota(device_type)
        qt.start_session(dt)
        print(f"Tracking {dt.upper()} quota: {qt.summary(dt)}")

    def stop_quota_tracking(self):
        """Stop and log usage."""
        qt, dt = self.quota()
        hours = qt.end_session()
        if hours:
            print(f"Session: {hours:.2f}h used. {qt.summary(dt)}")

    # ----------------------------------------------------------------
    # Notebook import
    # ----------------------------------------------------------------
    def run_notebook(self, path, stream=True, stop_on_error=True):
        """Load .ipynb and execute all code cells."""
        with open(path) as f:
            nb = json.load(f)
        cells = []
        for cell in nb.get("cells", []):
            if cell.get("cell_type") == "code":
                src = cell.get("source", "")
                if isinstance(src, list):
                    src = "".join(src)
                if src.strip():
                    cells.append(src)
        print(f"Loaded {len(cells)} cells from {path}")
        return self.execute_notebook(cells, stream=stream, stop_on_error=stop_on_error)

    # ----------------------------------------------------------------
    # Notifications
    # ----------------------------------------------------------------
    def execute_notify(self, code, notify_url=None, label="Execution", **kwargs):
        """Execute and send Slack/webhook notification on completion."""
        from kgz.notify import notify as _notify
        result = self.execute(code, **kwargs)
        if notify_url:
            status = "completed" if result.success else f"FAILED: {result.error_name}"
            _notify(notify_url, f"[kgz] {label} {status} ({result.elapsed_seconds:.1f}s)")
        return result

    # ----------------------------------------------------------------
    # Pipeline (run-once)
    # ----------------------------------------------------------------
    def pipeline(self, steps, notify_url=None, use_cache=False):
        """
        Run a pipeline: list of (label, code) tuples.
        Tracks quota, notifies, optionally caches.

        Returns list of (label, CellResult).
        """
        from kgz.notify import notify as _notify
        self.start_quota_tracking()
        results = []
        for i, (label, code) in enumerate(steps):
            print(f"\n--- [{i+1}/{len(steps)}] {label} ---")
            if use_cache:
                r = self.execute_cached(code, stream=True)
            else:
                r = self.execute(code, stream=True)
            results.append((label, r))
            if not r.success:
                _notify(notify_url, f"[kgz] Pipeline failed at '{label}': {r.error_name}")
                break
        self.stop_quota_tracking()
        passed = sum(1 for _, r in results if r.success)
        total_t = sum(r.elapsed_seconds for _, r in results)
        _notify(notify_url, f"[kgz] Pipeline: {passed}/{len(results)} steps ({total_t:.0f}s)")
        return results

    # ----------------------------------------------------------------
    # Model download with size reporting
    # ----------------------------------------------------------------
    def download_model(self, remote_path, local_path=None):
        """Download model file with size info."""
        from kgz.file_ops import download_file
        r = self.execute(f"import os; print(os.path.getsize('{remote_path}'))", stream=False)
        size = int(r.stdout.strip()) if r.success and r.stdout.strip().isdigit() else 0
        if size > 0:
            print(f"Downloading {remote_path} ({size/1024/1024:.1f} MB)...")
        local = local_path or os.path.basename(remote_path)
        download_file(self.base_url, remote_path, local)
        print(f"Saved: {local}")
        return local

    # ----------------------------------------------------------------
    # Environment snapshot / restore
    # ----------------------------------------------------------------
    def snapshot_env(self, path=None):
        """Save remote pip freeze locally."""
        r = self.execute("import subprocess; print(subprocess.check_output(['pip','freeze']).decode())",
                         stream=False)
        if not r.success:
            return None
        path = path or os.path.expanduser(f"~/.kgz/{self.name}-reqs.txt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(r.stdout)
        print(f"Snapshot: {path} ({len(r.stdout.splitlines())} packages)")
        return path

    def restore_env(self, path=None):
        """Restore pip packages from snapshot."""
        path = path or os.path.expanduser(f"~/.kgz/{self.name}-reqs.txt")
        if not os.path.exists(path):
            print(f"No snapshot: {path}")
            return
        from kgz.file_ops import upload_file
        upload_file(self.base_url, path, "_kgz_reqs.txt")
        self.execute("import subprocess,sys; subprocess.check_call("
                     "[sys.executable,'-m','pip','install','-q','-r','_kgz_reqs.txt'])",
                     stream=False, timeout=300)
        print(f"Restored from {path}")

    # ----------------------------------------------------------------
    # Parallel execution across multiple kernels
    # ----------------------------------------------------------------
    @staticmethod
    def parallel_execute(kernels, code, **kwargs):
        """Execute same code on multiple kernels in parallel."""
        import threading
        results = [None] * len(kernels)
        def _run(i, k):
            results[i] = k.execute(code, stream=False, **kwargs)
        threads = [threading.Thread(target=_run, args=(i, k)) for i, k in enumerate(kernels)]
        for t in threads: t.start()
        for t in threads: t.join()
        return results

    # ----------------------------------------------------------------
    # Health monitor
    # ----------------------------------------------------------------
    def monitor(self):
        """Get a KernelMonitor for health checks and progress parsing."""
        from kgz.health import KernelMonitor
        return KernelMonitor(self)

    def health_check(self):
        """Full health check with formatted dashboard."""
        return self.monitor().check_pretty()

    def training_progress(self):
        """Parse training metrics from latest execution output."""
        from kgz.health import parse_training_progress
        if self._history:
            output = self._history[-1].get("output", "")
            for line in reversed(output.strip().split("\n")):
                m = parse_training_progress(line)
                if m: return m
        return {}

    # ----------------------------------------------------------------
    # TPU-specific helpers
    # ----------------------------------------------------------------
    def is_tpu(self):
        """Check if kernel has TPU accelerator."""
        r = self.execute("import jax; print(jax.default_backend())", stream=False)
        return "tpu" in r.stdout.strip().lower()

    def device_info(self):
        """Get detailed device info (works for both GPU and TPU)."""
        r = self.execute("""
import jax
info = {"backend": jax.default_backend(), "device_count": jax.device_count(),
        "devices": [{"id": d.id, "kind": d.device_kind, "platform": d.platform}
                    for d in jax.devices()]}
try:
    info["process_count"] = jax.process_count()
    info["local_device_count"] = jax.local_device_count()
except: pass
import json; print(json.dumps(info))
""", stream=False)
        try:
            import json as _json
            return _json.loads(r.stdout.strip())
        except Exception:
            return {"raw": r.stdout}

    def tpu_type(self):
        """Get TPU type (e.g. 'TPU v3-8', 'TPU v2-8') or GPU type."""
        info = self.device_info()
        devices = info.get("devices", [])
        if devices:
            return devices[0].get("kind", "unknown")
        return info.get("backend", "unknown")

    # ----------------------------------------------------------------
    # Profiles
    # ----------------------------------------------------------------
    def save_profile(self, profile_name):
        """Save kernel config as reusable profile."""
        from kgz.profiles import save_profile
        config = {"url": self.base_url, "name": self.name}
        save_profile(profile_name, config)
        print(f"Profile '{profile_name}' saved")

    @classmethod
    def from_profile(cls, profile_name):
        """Create Kernel from saved profile."""
        from kgz.profiles import load_profile
        config = load_profile(profile_name)
        if not config: raise FileNotFoundError(f"No profile: {profile_name}")
        return cls(config["url"], name=config.get("name", profile_name))

    # ----------------------------------------------------------------
    # Audit
    # ----------------------------------------------------------------
    def _audit(self, action, details=None):
        from kgz.audit import log_action
        log_action(action, self.name, details)

    # ----------------------------------------------------------------
    # Budget (quota-based)
    # ----------------------------------------------------------------
    def set_budget(self, max_hours, notify_url=None):
        """
        Monitor quota and alert when approaching limit.

        Args:
            max_hours: Max hours to use this session
            notify_url: Slack/webhook for alerts
        """
        from kgz.notify import notify as _notify
        import time as _time
        qt, dt = self.quota()
        qt.start_session(dt)
        alert_at = max_hours * 0.8
        alerted = False
        print(f"Budget: {max_hours}h (alert at {alert_at:.1f}h)")

        while True:
            used = qt.used_this_week(dt)
            session_used = (_time.time() - qt._session_start) / 3600 if qt._session_start else 0
            if session_used >= max_hours:
                msg = f"[kgz] Budget exceeded: {session_used:.1f}h used (limit: {max_hours}h)"
                print(msg)
                _notify(notify_url, msg)
                self.interrupt()
                break
            if session_used >= alert_at and not alerted:
                msg = f"[kgz] Budget alert: {session_used:.1f}/{max_hours}h"
                print(msg)
                _notify(notify_url, msg)
                alerted = True

            if self.status() == "idle":
                break
            _time.sleep(60)

        qt.end_session()

    # ----------------------------------------------------------------
    # Kaggle dataset helpers
    # ----------------------------------------------------------------
    def attach_dataset(self, dataset_slug):
        """
        Make a Kaggle dataset available in the kernel.
        Kaggle datasets are pre-mounted at /kaggle/input/{dataset_name}.

        Args:
            dataset_slug: e.g. "openai/gsm8k" or "username/dataset-name"
        """
        name = dataset_slug.split("/")[-1]
        r = self.execute(f"""
import os
path = f"/kaggle/input/{name}"
if os.path.exists(path):
    files = os.listdir(path)
    print(f"Dataset mounted: {{path}} ({{len(files)}} files)")
    for f in files[:10]:
        size = os.path.getsize(os.path.join(path, f))
        print(f"  {{f}}: {{size/1024/1024:.1f}} MB")
else:
    print(f"Dataset not found: {{path}}")
    print("Available datasets:")
    if os.path.exists("/kaggle/input"):
        for d in os.listdir("/kaggle/input"):
            print(f"  /kaggle/input/{{d}}")
""".replace("{name}", name), stream=False)
        return r

    def list_datasets(self):
        """List all mounted Kaggle datasets."""
        r = self.execute("""
import os
if os.path.exists("/kaggle/input"):
    for d in sorted(os.listdir("/kaggle/input")):
        path = f"/kaggle/input/{d}"
        files = os.listdir(path) if os.path.isdir(path) else []
        print(f"  {d}: {len(files)} files")
else:
    print("No datasets mounted")
""", stream=False)
        print(r.stdout)
        return r

    # ----------------------------------------------------------------
    # Convenience
    # ----------------------------------------------------------------
    def run(self, code, **kwargs):
        """Alias for execute()."""
        return self.execute(code, **kwargs)

    @property
    def history(self):
        return self._history.copy()

    def __repr__(self):
        return f"Kernel(name={self.name!r}, id={self.kernel_id!r})"

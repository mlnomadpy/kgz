---
layout: home
title: kgz
---

# kgz

Execute code on Kaggle GPU & TPU kernels from your terminal. Free compute, no browser.

```bash
pip install kgz
```

```python
from kgz import Kernel

k = Kernel(url)
result = k.execute("import jax; print(jax.devices())", stream=False)
print(result.stdout)  # [CudaDevice(id=0), CudaDevice(id=1)]
```

---


# Complete Usage Guide

## Installation

```bash
pip install kgz
```

Single dependency: `websocket-client`.

## Getting Your Kaggle URL

1. Open a notebook on [kaggle.com/code](https://www.kaggle.com/code)
2. Enable **GPU** or **TPU** in Settings
3. Copy the URL from your browser:
   ```
   https://kkb-production.jupyter-proxy.kaggle.net/k/12345/eyJhbG.../proxy
   ```
4. Pass it to kgz

The URL contains a session JWT. It expires when the session ends (~12h GPU, ~9h TPU).

## Core: Connect & Execute

```python
from kgz import Kernel

k = Kernel(url)
k = Kernel(url, name="my-session")  # Named for save/resume
```

### Execute Code

```python
result = k.execute("print('hello')", stream=False)

result.success          # True
result.stdout           # "hello\n"
result.return_value     # None (print has no return)
result.error_name       # None
result.elapsed_seconds  # 0.1
```

**Important:** Use `stream=False` in scripts. `stream=True` (default) prints to stdout.

### Expressions Return Values

```python
result = k.execute("2 + 2", stream=False)
result.return_value  # "4"
```

### Error Handling

```python
result = k.execute("1/0", stream=False)
result.success     # False
result.error_name  # "ZeroDivisionError"
result.error_value # "division by zero"

# Or raise exceptions
from kgz import KernelError
try:
    k.execute("1/0", raise_on_error=True, stream=False)
except KernelError as e:
    print(e.result.traceback)
```

## Quota Tracking

Kaggle limits: **30h/week GPU**, **20h/week TPU**.

```python
k.start_quota_tracking()     # Start counting
# ... work ...
k.stop_quota_tracking()      # Log usage

k.quota_summary()
# GPU: 2.1h used / 30h quota (27.9h remaining)
# Session limit: 9.9h remaining
```

## Output Caching

Cache idempotent cells (pip install, imports, data loading):

```python
result = k.execute_cached("import jax; print(jax.__version__)")
# First call: 100ms (remote execution)
# Second call: 1ms (local cache) — 98x speedup
```

Errors are not cached — only successful results.

```python
k.clear_cache()  # Reset
```

## Pipeline

Run a sequence of steps with quota tracking, caching, and notifications:

```python
results = k.pipeline([
    ("Install", "pip install -q jax flax"),
    ("Check GPU", "import jax; print(jax.devices())"),
    ("Load data", "data = load_dataset('tiny')"),
    ("Train", "train(data, steps=5000)"),
], notify_url="https://hooks.slack.com/...", use_cache=True)

# Setup steps are cached, training is not
# Slack notification on completion or failure

for label, result in results:
    print(f"{label}: {'OK' if result.success else 'FAIL'}")
```

## Multi-Cell Execution

Execute cells sequentially with shared state:

```python
results = k.execute_notebook([
    "import jax",
    "model = build()",
    "loss = train(model)",
], stop_on_error=True, stream=False)
```

## Import & Run .ipynb

```python
results = k.run_notebook("training.ipynb", stop_on_error=True)
```

## Inspect Remote State

### Variable Snapshot

```python
snap = k.snapshot()
# {"model": {"type": "GPT", "shape": "(18M,)", "repr": "GPT(...)"},
#  "loss": {"type": "float", "repr": "2.31"},
#  "data": {"type": "ndarray", "shape": "(106212345,)", "len": 106212345}}
```

### GPU/TPU Resources

```python
res = k.resources()
# {"backend": "gpu", "device_count": 2,
#  "gpus": [{"utilization": 85, "memory_used_mb": 12000, "memory_total_mb": 15360}],
#  "cpu_percent": 17.2, "ram_used_gb": 8.6, "ram_total_gb": 33.7}
```

## File Operations

### Upload / Download

```python
from kgz import upload_file, download_file
from kgz.file_ops import upload_directory, list_files

upload_file(url, "model.py", "model.py")
upload_directory(url, "./src", "src")
download_file(url, "/kaggle/working/results.json", "./results.json")
```

### Model Download (with size)

```python
k.download_model("/kaggle/working/model.pkl", "./model.pkl")
# Downloading model.pkl (73.2 MB)...
# Saved: ./model.pkl
```

### File Sync (Watch Mode)

```python
from kgz import FileSync

sync = FileSync(url, "./src", "/kaggle/working/src")
sync.push()      # One-shot upload of changed files
sync.start()     # Background watcher — auto-uploads on change
# ... edit files locally ...
sync.stop()
```

## Environment

### Secrets

```python
k.set_env(HF_TOKEN="hf_...", WANDB_API_KEY="...")
# Secrets are excluded from execution history and notebook export
```

### Snapshot / Restore

```python
k.snapshot_env()    # pip freeze → ~/.kgz/{name}-reqs.txt
k.restore_env()     # pip install from snapshot
```

## Notifications

```python
k.execute_notify(code,
    notify_url="https://hooks.slack.com/...",
    label="Training")
# Sends Slack message: "[kgz] Training completed (45.2s)"
```

## Session Persistence

```python
k.save_session()                     # Save to ~/.kgz/{name}.json
k = Kernel.resume("my-session")     # Resume in a new script
Kernel.list_sessions()               # List all saved
```

## Notebook Export

```python
k.to_notebook("output.ipynb")
# Exports execution history as a real Jupyter notebook
```

## Parallel Execution

Run on multiple Kaggle sessions simultaneously:

```python
k1 = Kernel(url1)
k2 = Kernel(url2)
results = Kernel.parallel_execute([k1, k2],
    "import jax; print(jax.devices())")
```

## CLI

```bash
kgz run URL "code"              # Execute code
kgz exec URL -f script.py       # Execute local file
kgz status URL                  # idle/busy
kgz interrupt URL               # Ctrl-C
kgz wait URL                    # Block until idle
kgz restart URL                 # Restart kernel
kgz upload URL file [remote]    # Upload file
kgz download URL remote [local] # Download file
kgz ls URL [path]               # List files
kgz info URL                    # Kernel info
kgz snapshot URL                # Variable inspection
kgz resources URL               # GPU/CPU usage
kgz sync URL local_dir          # Watch & sync
kgz notebook URL -f cells.txt   # Run notebook
kgz sessions                    # List saved sessions
```

## Kaggle Limits

| Resource | Limit |
|----------|-------|
| GPU | 30 hours/week |
| TPU | 20 hours/week |
| GPU session | 12 hours max |
| TPU session | 9 hours max |
| Disk | 73 GB |
| RAM | 13-30 GB (depends on GPU) |

Use `k.quota_summary()` to check remaining time.

## GPU & TPU Detection

kgz works with both Kaggle GPU (T4) and TPU (v3-8) accelerators:

```python
k.is_tpu()         # True if TPU kernel
k.tpu_type()       # "TPU v3-8" or "Tesla T4"
k.device_info()    # Full details: backend, device_count, platform per device
```

## Health Dashboard

```python
k.health_check()
#   Kaggle Kernel Health
#   ==================================================
#   Kernel:  busy
#   Backend: gpu (2 devices)
#   GPU 0:   85% util, 12000/15360 MB
#   GPU 1:   82% util, 11500/15360 MB
#   CPU:     17%
#   RAM:     8.6/33.7 GB
#   Train:   step 1234/5000 | loss 2.31 | 56,000 tok/s
#   ETA:     ~35m
#   Quota:   27.9h remaining (GPU)
#   Session: 9.8h before expiry
```

### Training Progress Parsing

kgz automatically parses metrics from your training output:

```python
progress = k.training_progress()
# {"step": 1234, "total_steps": 5000, "loss": 2.31, "tok_per_sec": 56000}
```

Supports common log formats from JAX, PyTorch, flaxchat, nanochat.

## Budget Alerts

Kaggle quota is limited. Set a budget to avoid running out:

```python
k.set_budget(max_hours=8, notify_url="https://hooks.slack.com/...")
# Alerts at 6.4h (80%), interrupts at 8h
```

## Kaggle Datasets

Kaggle datasets are pre-mounted at `/kaggle/input/`:

```python
k.list_datasets()
#   gsm8k: 3 files
#   tiny-shakespeare: 1 files

k.attach_dataset("openai/gsm8k")
# Dataset mounted: /kaggle/input/gsm8k (3 files)
#   train.jsonl: 12.4 MB
#   test.jsonl: 1.2 MB
```

## Profiles

Save kernel configurations for reuse across sessions:

```python
k.save_profile("gpu-training")

# Later, in a new script:
k = Kernel.from_profile("gpu-training")
```

## Audit Log

Every action is logged for debugging:

```python
from kgz.audit import print_history
print_history()
# 2026-04-04 10:23  my-session  execute
# 2026-04-04 10:24  my-session  execute_cached
# 2026-04-04 10:25  my-session  interrupt
```

---


# kgz — Agent Integration Guide

This document is designed for AI coding agents (Claude Code, Cursor, Copilot, etc.) to understand and use kgz effectively.

## What kgz Does

kgz connects to a running Kaggle Jupyter notebook via WebSocket and lets you execute Python code remotely. Kaggle provides free GPUs (2x T4, 16GB each) and TPUs (v3-8).

## Quick Reference

```python
from kgz import Kernel

k = Kernel(url)                          # Connect (auto-discovers kernel)
result = k.execute(code)                 # Execute code → CellResult
result = k.run(code)                     # Alias for execute()
k.status()                               # 'idle' | 'busy'
k.interrupt()                            # Stop execution
k.wait()                                 # Block until idle
k.restart()                              # Restart kernel (clears state)
k.close()                                # Close connection
```

## CellResult — Structured Output

Every `execute()` returns a `CellResult`:

```python
result = k.execute("print('hello'); x = 42")
result.success        # bool — True if no exception
result.stdout         # str — captured print() output
result.stderr         # str — captured stderr
result.return_value   # str | None — last expression value (like Jupyter Out[])
result.error_name     # str | None — exception class name
result.error_value    # str | None — exception message
result.traceback      # list[str] — traceback lines
result.elapsed_seconds # float — wall time
result.output         # str — stdout + return_value combined
```

### Decision Pattern for Agents

```python
result = k.execute(code, stream=False)
if result.success:
    # Parse result.stdout or result.return_value
    data = result.stdout.strip()
else:
    # Handle error
    print(f"Failed: {result.error_name}: {result.error_value}")
    # Maybe retry with fixed code
```

### Raise on Error

```python
from kgz import KernelError

try:
    k.execute(code, raise_on_error=True, stream=False)
except KernelError as e:
    print(e.result.traceback)  # Full traceback for debugging
```

## Inspecting Remote State

```python
# See all variables (names, types, shapes)
snapshot = k.snapshot()
# Returns: {"model": {"type": "GPT", "repr": "GPT(...)"}, "loss": {"type": "float"}, ...}

# Check GPU/TPU usage
resources = k.resources()
# Returns: {"backend": "gpu", "device_count": 2, "gpus": [{"utilization": 85, "memory_used_mb": 12000}], ...}
```

## Multi-Cell Pipelines

Execute cells sequentially with shared state (like running a notebook):

```python
results = k.execute_notebook([
    "import jax",
    "model = build_model()",
    "loss = train(model)",
], stop_on_error=True, stream=False)

# Check each cell
for i, r in enumerate(results):
    if not r.success:
        print(f"Cell {i} failed: {r.error_name}")
        break
```

## File Transfer

```python
from kgz import upload_file, download_file
from kgz.file_ops import list_files, upload_directory

# Upload a single file
upload_file(url, "model.py", "model.py")

# Upload entire directory
upload_directory(url, "./src", "src")

# Download results
download_file(url, "/kaggle/working/results.json", "./results.json")

# List remote files
files = list_files(url)  # [{"name": "...", "type": "file", "size": 123}, ...]
```

## File Sync (Watch Mode)

```python
from kgz import FileSync

sync = FileSync(url, "./src", "/kaggle/working/src")
sync.push()      # One-shot upload of changed files
sync.start()     # Background watch + auto-upload
# ... edit files locally, they appear on Kaggle ...
sync.stop()
```

## Environment Variables

Set secrets without them appearing in execution history:

```python
k.set_env(HF_TOKEN="hf_...", WANDB_API_KEY="...")
```

## Session Persistence

```python
# Save session for later
k.save_session()  # Saves to ~/.kgz/{name}.json

# Resume later (even after restarting your script)
k = Kernel.resume("session-name")

# List all saved sessions
sessions = Kernel.list_sessions()
```

## Export to Notebook

```python
# Export execution history as .ipynb
k.to_notebook("output.ipynb")
```

## Connection Management

```python
# Persistent connection (reuses WebSocket)
k = Kernel(url)
k.execute("x = 1")    # Uses existing WS
k.execute("print(x)")  # Same WS, no reconnect

# Auto-reconnect on disconnect (3 retries)
k.execute("long_running()")  # Reconnects if WS drops

# Context manager
with Kernel(url) as k:
    k.execute("train()")
# Auto-closes on exit
```

## CLI Commands

```bash
kgz run URL "code"              # Execute code
kgz exec URL -f script.py       # Execute file
kgz status URL                  # idle/busy
kgz interrupt URL               # Ctrl-C
kgz wait URL                    # Block until idle
kgz restart URL                 # Restart kernel
kgz upload URL file [remote]    # Upload file
kgz download URL remote [local] # Download file
kgz ls URL [path]               # List files
kgz info URL                    # Kernel info
kgz snapshot URL                # Variable inspection
kgz resources URL               # GPU/CPU usage
kgz sync URL local_dir          # Watch & sync
kgz notebook URL -f cells.txt   # Run notebook
kgz sessions                    # List sessions
```

## Common Patterns for AI Agents

### 1. Check Environment First

```python
k = Kernel(url)
res = k.resources()
if res.get("device_count", 0) >= 2:
    print("Multi-GPU available")
```

### 2. Install Dependencies

```python
k.execute("import subprocess, sys; subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'jax', 'flax'])")
```

### 3. Upload Code + Run

```python
from kgz.file_ops import upload_directory
upload_directory(url, "./my_project", "my_project")
k.execute("import sys; sys.path.insert(0, '.'); from my_project import train; train()")
```

### 4. Monitor Long Training

```python
k.execute("train(steps=5000)", stream=True)  # Streams output live
# Or async:
k.execute("train(steps=5000)", stream=False)  # Returns when done
```

### 5. Recover from Errors

```python
result = k.execute(code, stream=False)
if not result.success and "CUDA out of memory" in result.error_value:
    k.execute("import gc; gc.collect(); import jax; jax.clear_caches()")
    # Retry with smaller batch
```

---

[GitHub](https://github.com/mlnomadpy/kgz) · [PyPI](https://pypi.org/p/kgz) · [tpuz](https://github.com/mlnomadpy/tpuz)

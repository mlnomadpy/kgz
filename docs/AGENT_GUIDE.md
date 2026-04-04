---
layout: page
title: Agent Guide
permalink: /agent-guide/
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

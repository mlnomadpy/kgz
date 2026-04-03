# kgz — Execute Code on Kaggle Jupyter Kernels

Use this knowledge when the user wants to run code on Kaggle, use Kaggle GPUs/TPUs remotely, or when you see `import kgz` or `from kgz import`.

## Install

```bash
pip install kgz
```

## Core API

```python
from kgz import Kernel, CellResult, KernelError

# Connect (user provides URL from their Kaggle notebook browser tab)
k = Kernel("https://kkb-production.jupyter-proxy.kaggle.net/k/.../proxy")

# Execute code — returns structured CellResult
result = k.execute("print('hello')", stream=False)
result.success        # bool
result.stdout         # str — print() output
result.return_value   # str | None — last expression (like Jupyter Out[])
result.error_name     # str | None — exception class
result.error_value    # str | None — exception message
result.traceback      # list — traceback lines
result.elapsed_seconds # float

# Aliases
result = k.run(code, stream=False)  # same as execute()

# Control
k.status()      # 'idle' or 'busy'
k.interrupt()   # Ctrl-C
k.wait()        # Block until idle
k.restart()     # Restart kernel (clears state)
k.is_alive()    # True if reachable
k.close()       # Close persistent WebSocket
```

## IMPORTANT: Always use stream=False from code

`stream=True` (default) prints to stdout. When calling from agent code, always pass `stream=False` and read `result.stdout` instead.

## Error Handling

```python
# Check success
result = k.execute(code, stream=False)
if not result.success:
    print(f"Error: {result.error_name}: {result.error_value}")

# Or raise exceptions
try:
    k.execute(code, stream=False, raise_on_error=True)
except KernelError as e:
    print(e.result.traceback)
```

## Multi-Cell Pipeline

```python
results = k.execute_notebook([
    "import jax",
    "model = build()",
    "train(model)",
], stop_on_error=True, stream=False)

for i, r in enumerate(results):
    print(f"Cell {i}: {'OK' if r.success else r.error_name}")
```

## Inspect Remote State

```python
# Variable snapshot
snap = k.snapshot()
# {"model": {"type": "GPT", "shape": "..."}, "loss": {"type": "float"}, ...}

# GPU/TPU resources
res = k.resources()
# {"backend": "gpu", "device_count": 2, "gpus": [{"utilization": 85, "memory_used_mb": 12000}]}
```

## File Operations

```python
from kgz import upload_file, download_file
from kgz.file_ops import list_files, upload_directory

upload_file(url, "model.py", "model.py")
upload_directory(url, "./src", "src")
download_file(url, "/kaggle/working/results.json", "./results.json")
files = list_files(url)
```

## File Sync (Watch Mode)

```python
from kgz import FileSync

sync = FileSync(url, "./src", "/kaggle/working/src")
sync.push()      # One-shot upload changed files
sync.start()     # Background watch thread
sync.stop()
```

## Environment Variables (secrets)

```python
k.set_env(HF_TOKEN="hf_...", WANDB_API_KEY="...")
# Set without appearing in execution history
```

## Session Persistence

```python
k.save_session()                    # Save to ~/.kgz/{name}.json
k = Kernel.resume("session-name")   # Resume later
Kernel.list_sessions()               # List all saved
```

## Notebook Export

```python
k.to_notebook("output.ipynb")  # Export execution history as .ipynb
```

## CLI

```bash
kgz run URL "code"
kgz exec URL -f script.py
kgz status URL
kgz interrupt URL
kgz wait URL
kgz upload URL file.py
kgz download URL remote.json ./local.json
kgz ls URL
kgz snapshot URL
kgz resources URL
kgz sync URL ./local_dir
kgz sessions
```

## Common Patterns

### Install deps then train
```python
k.execute("import subprocess,sys; subprocess.check_call([sys.executable,'-m','pip','install','-q','jax','flax'])", stream=False)
k.execute(open("train.py").read(), timeout=3600, stream=True)
```

### Check GPU before running
```python
res = k.resources()
if res.get("device_count", 0) >= 2:
    k.execute("train(batch_size=64)")  # multi-GPU
else:
    k.execute("train(batch_size=32)")  # single GPU
```

### Recover from OOM
```python
r = k.execute(code, stream=False)
if not r.success and "out of memory" in str(r.error_value).lower():
    k.execute("import gc; gc.collect()")
    k.execute(code_with_smaller_batch, stream=False)
```

## Source

- Repo: `/Users/tahabsn/Documents/GitHub/kgz`
- 1,477 lines, 37 offline tests + 14 live tests
- Zero deps except `websocket-client`

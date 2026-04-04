# kgz — Execute Code on Kaggle Kernels (GPU & TPU)

Use when user wants to run code on Kaggle, use Kaggle GPUs/TPUs, or sees `import kgz`.

## Install

```bash
pip install kgz
```

## Core

```python
from kgz import Kernel
k = Kernel(url)
k = Kernel(url, name="session")

result = k.execute(code, stream=False)  # ALWAYS stream=False from agents
result.success / result.stdout / result.return_value / result.error_name

k.status()        # 'idle' | 'busy'
k.interrupt()     # Ctrl-C
k.wait()          # Block until idle
k.restart()       # Reset kernel
k.close()         # Close WebSocket
```

## GPU & TPU Support

```python
k.is_tpu()          # True if TPU kernel
k.device_info()     # Full device details (GPU/TPU)
k.tpu_type()        # "TPU v3-8" or "Tesla T4"
k.resources()       # GPU util%, VRAM, CPU%, RAM
```

## Health Monitor

```python
k.health_check()        # Full dashboard: kernel, GPU/TPU, training metrics, quota
k.training_progress()   # Parse step/loss/lr/tok_s from output
k.monitor().check()     # Programmatic health status dict
```

## Quota (GPU: 30h/week, TPU: 20h/week)

```python
k.quota_summary()           # Remaining hours + session time
k.start_quota_tracking()    # Start counting
k.stop_quota_tracking()     # Log usage
k.set_budget(max_hours=8, notify_url=slack)  # Alert + interrupt at limit
```

## Caching (98x speedup on repeated calls)

```python
result = k.execute_cached(code)  # Second call: from local cache
k.clear_cache()
```

## Pipeline

```python
results = k.pipeline([
    ("Install", "pip install -q jax"),
    ("Train", "train(steps=5000)"),
], notify_url=slack, use_cache=True)
```

## Notebook Import/Export

```python
results = k.run_notebook("training.ipynb")
k.to_notebook("output.ipynb")
```

## Files

```python
from kgz import upload_file, download_file, FileSync
upload_file(url, "model.py", "model.py")
download_file(url, "/kaggle/working/model.pkl", "./")
k.download_model("/kaggle/working/model.pkl")  # With size reporting
sync = FileSync(url, "./src", "/kaggle/working/src")
sync.start()  # Background watch
```

## Kaggle Datasets

```python
k.attach_dataset("openai/gsm8k")   # Check if mounted
k.list_datasets()                    # List all mounted datasets
```

## Secrets (excluded from history)

```python
k.set_env(WANDB_API_KEY="...", HF_TOKEN="...")
```

## Environment

```python
k.snapshot_env()     # pip freeze → local file
k.restore_env()      # Restore packages
```

## Sessions & Profiles

```python
k.save_session()                 # Save for resume
k = Kernel.resume("name")       # Resume later
k.save_profile("gpu-training")  # Save config
k = Kernel.from_profile("gpu-training")
```

## Parallel

```python
results = Kernel.parallel_execute([k1, k2], code)
```

## Notifications

```python
k.execute_notify(code, notify_url=slack, label="Training")
```

## CLI

```bash
kgz run/exec/status/interrupt/wait/restart
kgz upload/download/ls/sync
kgz snapshot/resources/sessions/notebook
```

## Kaggle Limits

GPU: 30h/week, 12h/session. TPU: 20h/week, 9h/session.

## Source

- 2,000+ lines, 54 offline + 14 live tests
- Dep: websocket-client

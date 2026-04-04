<div align="center">

# kgz

**Execute code on Kaggle GPU & TPU kernels from your terminal.**

Free GPUs. Free TPUs. No browser needed.

[![PyPI](https://img.shields.io/pypi/v/kgz)](https://pypi.org/project/kgz/)
[![Tests](https://img.shields.io/github/actions/workflow/status/mlnomadpy/kgz/test.yaml?label=tests)](https://github.com/mlnomadpy/kgz/actions)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)

[Guide](https://www.tahabouhsine.com/kgz/guide/) · [Agent Guide](https://www.tahabouhsine.com/kgz/agent-guide/) · [GitHub](https://github.com/mlnomadpy/kgz)

</div>

---

## Why?

Kaggle gives you **free GPUs** (2x T4, 30h/week) and **free TPUs** (v3-8, 20h/week). But you're stuck in a browser notebook. kgz lets you drive Kaggle from your terminal, scripts, or AI agents.

```python
from kgz import Kernel

k = Kernel("https://kkb-production.jupyter-proxy.kaggle.net/k/.../proxy")
result = k.execute("import jax; print(jax.devices())")
# [CudaDevice(id=0), CudaDevice(id=1)]
```

## Install

```bash
pip install kgz
```

Single dependency: `websocket-client`.

## Getting Your URL

1. Open a notebook on [kaggle.com/code](https://www.kaggle.com/code)
2. Enable **GPU** or **TPU** in Settings
3. Copy the URL from your browser bar
4. Pass it to kgz

## Features

### Execute with Structured Output

```python
result = k.execute("print('hello'); 2+2", stream=False)
result.success        # True
result.stdout         # "hello\n"
result.return_value   # "4"
result.elapsed_seconds # 0.1
```

### GPU & TPU Support

```python
k.is_tpu()            # True if TPU kernel
k.device_info()       # Full device details
k.tpu_type()          # "TPU v3-8" or "Tesla T4"
k.resources()         # GPU util%, VRAM, CPU%, RAM
```

### Health Dashboard

```python
k.health_check()
#   Kernel:  busy
#   Backend: gpu (2 devices)
#   GPU 0:   85% util, 12000/15360 MB
#   GPU 1:   82% util, 11500/15360 MB
#   CPU:     17%
#   RAM:     8.6/33.7 GB
#   Train:   step 1234/5000 | loss 2.31 | 56,000 tok/s
#   ETA:     ~35m
#   Quota:   27.9h remaining (GPU)
```

### Quota Tracking (30h/week GPU, 20h/week TPU)

```python
k.start_quota_tracking()
# ... train for hours ...
k.stop_quota_tracking()
k.quota_summary()
# GPU: 4.2h used / 30h quota (25.8h remaining)
# Session: 7.8h before expiry
```

### Output Caching (98x speedup)

```python
result = k.execute_cached("pip install -q jax flax")
# First: 5s (remote), Second: 0.001s (local cache)
```

### Pipeline (Run-Once)

```python
results = k.pipeline([
    ("Install", "pip install -q jax flax"),
    ("Check GPU", "import jax; print(jax.devices())"),
    ("Train", "train(steps=5000)"),
], notify_url="https://hooks.slack.com/...", use_cache=True)
# Tracks quota, caches setup, notifies Slack on complete/fail
```

### Notebook Import & Export

```python
results = k.run_notebook("training.ipynb")   # Run existing .ipynb
k.to_notebook("output.ipynb")                # Export history as .ipynb
```

### Kaggle Datasets

```python
k.list_datasets()                        # Show mounted datasets
k.attach_dataset("openai/gsm8k")        # Check availability
```

### File Operations

```python
from kgz import upload_file, download_file, FileSync

upload_file(url, "model.py", "model.py")
k.download_model("/kaggle/working/model.pkl", "./model.pkl")

sync = FileSync(url, "./src", "/kaggle/working/src")
sync.start()   # Background watch — auto-upload on change
```

### Secrets (excluded from history & export)

```python
k.set_env(WANDB_API_KEY="...", HF_TOKEN="...")
```

### Environment Snapshot

```python
k.snapshot_env()    # pip freeze → local file
k.restore_env()     # Restore packages (after kernel restart)
```

### Budget Alerts

```python
k.set_budget(max_hours=8, notify_url="https://hooks.slack.com/...")
# Alert at 80%, interrupt at limit
```

### Sessions & Profiles

```python
k.save_session()                          # Save for later
k = Kernel.resume("my-session")          # Resume

k.save_profile("gpu-training")           # Save config
k = Kernel.from_profile("gpu-training")  # Reuse
```

### Notifications

```python
k.execute_notify(code, notify_url="https://hooks.slack.com/...", label="Training")
```

### Parallel Execution

```python
k1, k2 = Kernel(url1), Kernel(url2)
results = Kernel.parallel_execute([k1, k2], "import jax; print(jax.devices())")
```

### Persistent Connection

WebSocket reused across calls — no reconnect overhead:

```python
for i in range(100):
    k.execute(f"step({i})", stream=False)  # ~100ms each, not 500ms
```

### Error Handling

```python
from kgz import KernelError
try:
    k.execute("1/0", raise_on_error=True, stream=False)
except KernelError as e:
    print(e.result.traceback)
```

## CLI

```bash
kgz run URL "code"              # Execute code
kgz exec URL -f script.py       # Execute file
kgz status URL                  # idle/busy
kgz interrupt URL               # Ctrl-C
kgz wait URL                    # Block until idle
kgz upload URL file [remote]    # Upload
kgz download URL remote [local] # Download
kgz ls URL [path]               # List files
kgz snapshot URL                # Variable inspection
kgz resources URL               # GPU/TPU/CPU usage
kgz sync URL local_dir          # Watch & sync
kgz notebook URL -f cells.txt   # Run notebook cells
kgz sessions                    # List saved sessions
```

## Kaggle Limits

| Resource | Weekly Limit | Session Max |
|----------|-------------|-------------|
| GPU (T4) | 30 hours | 12 hours |
| TPU (v3-8) | 20 hours | 9 hours |

Use `k.quota_summary()` to check remaining time.

## Documentation

- **[Usage Guide](https://www.tahabouhsine.com/kgz/guide/)** — Complete feature walkthrough
- **[Agent Guide](https://www.tahabouhsine.com/kgz/agent-guide/)** — Patterns for AI coding agents

## Pair with tpuz

```bash
pip install kgz     # Kaggle free GPUs/TPUs
pip install tpuz    # GCP TPU/GPU pods
```

## Claude Code

```bash
mkdir -p ~/.claude/skills/kgz-guide
cp SKILL.md ~/.claude/skills/kgz-guide/skill.md
```

## License

MIT

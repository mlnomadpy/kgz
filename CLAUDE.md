# kgz — Claude Code Integration

## What This Is

kgz executes code on Kaggle Jupyter kernels (GPU & TPU) via WebSocket.

## Key Modules

| Module | What |
|--------|------|
| `kernel.py` | Core: execute, cache, pipeline, health, quota, TPU support, profiles |
| `file_ops.py` | Upload/download via Jupyter Contents API |
| `sync.py` | Background file sync with change detection |
| `cache.py` | Local result caching (98x speedup) |
| `quota.py` | GPU/TPU weekly quota tracking |
| `health.py` | Training progress parser, health monitor |
| `profiles.py` | Save/load kernel configs |
| `audit.py` | Action history |
| `notify.py` | Slack/webhook notifications |

## Quick Usage

```python
from kgz import Kernel
k = Kernel(url, name="my-session")
result = k.execute(code, stream=False)  # ALWAYS stream=False
k.health_check()       # Dashboard
k.quota_summary()      # Remaining hours
k.is_tpu()             # True if TPU
```

## IMPORTANT: stream=False

Always use `stream=False` from agent code. Default `stream=True` prints to stdout.

## Tests

```bash
pytest tests/ -k "not live" -v  # 54 tests
```

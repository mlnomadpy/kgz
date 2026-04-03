# kgz — Claude Code Integration

## What This Is

kgz is a Python library for executing code on remote Kaggle Jupyter kernels via WebSocket. It lets you offload GPU/TPU computation to Kaggle's free hardware.

## Key Files

- `kgz/kernel.py` — Core `Kernel` class (persistent WS, structured output, auto-retry)
- `kgz/file_ops.py` — Upload/download files via Jupyter Contents API
- `kgz/sync.py` — Background file sync with change detection
- `kgz/cli.py` — CLI with 14 commands

## How To Use

```python
from kgz import Kernel

k = Kernel("https://kkb-production.jupyter-proxy.kaggle.net/k/.../proxy")
result = k.execute("print('hello')", stream=False)
# result.success = True, result.stdout = "hello\n"
```

## Important: stream=False for Agents

Always use `stream=False` when calling from code. `stream=True` (default) prints to stdout which clutters agent output.

## Checking If Code Worked

```python
r = k.execute(code, stream=False)
if r.success:
    output = r.stdout  # or r.return_value for expressions
else:
    error = f"{r.error_name}: {r.error_value}"
```

## Running Tests

```bash
# Offline tests (no Kaggle needed)
pytest tests/ -k "not live and not Live"

# Live tests (needs running Kaggle session)
KGZ_TEST_URL="https://kkb-..." pytest tests/ -v
```

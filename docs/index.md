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

k.health_check()       # GPU util, training progress, quota
k.quota_summary()      # 27.9h remaining
```

## Documentation

- **[Complete Guide](guide.md)** — Every feature with examples
- **[Agent Guide](AGENT_GUIDE.md)** — Integration patterns for AI coding agents

## Features

| Category | Features |
|----------|----------|
| **Execute** | `execute`, `execute_cached` (98x speedup), `pipeline`, `run_notebook` |
| **GPU/TPU** | `is_tpu`, `device_info`, `tpu_type`, `resources` |
| **Monitor** | `health_check`, `training_progress`, `quota_summary` |
| **Files** | `upload_file`, `download_model`, `FileSync` (watch mode) |
| **Session** | `save_session`, `resume`, `save_profile`, `to_notebook` |
| **Budget** | `set_budget`, `start_quota_tracking`, notifications |
| **Data** | `list_datasets`, `attach_dataset` |
| **Parallel** | `parallel_execute` across multiple kernels |

## Kaggle Limits

| Resource | Weekly | Session |
|----------|--------|---------|
| GPU (2x T4) | 30h | 12h |
| TPU (v3-8) | 20h | 9h |

[GitHub](https://github.com/mlnomadpy/kgz) · [PyPI](https://pypi.org/p/kgz) · [tpuz](https://github.com/mlnomadpy/tpuz)

# kgz

Execute code on remote Kaggle Jupyter kernels from your terminal.

```bash
pip install kgz
```

## Why?

Kaggle gives you **free GPUs** (2x T4) and **free TPUs** (v3-8). But you're stuck in a browser notebook. kgz lets you drive Kaggle kernels from your local machine — your editor, your scripts, your workflow.

## Quick Start

### Python

```python
from kgz import Kernel

# Paste your Kaggle notebook URL
k = Kernel("https://kkb-production.jupyter-proxy.kaggle.net/k/.../proxy")

# Execute code remotely
k.run("import jax; print(jax.devices())")
# [CudaDevice(id=0), CudaDevice(id=1)]

# Check status
k.status()  # 'idle'

# Run a training script
k.run(open("train.py").read(), timeout=3600)

# Wait for it
k.wait()
```

### CLI

```bash
# Run code
kgz run URL "import torch; print(torch.cuda.device_count())"

# Execute a local script
kgz exec URL -f train.py

# Check status
kgz status URL

# Interrupt
kgz interrupt URL

# Upload/download files
kgz upload URL model.py
kgz download URL /kaggle/working/results.json ./results.json

# List remote files
kgz ls URL

# Wait for completion
kgz wait URL --timeout 3600
```

## How It Works

kgz connects to the Jupyter kernel running inside your Kaggle notebook via WebSocket. When you open a Kaggle notebook, the URL contains a JWT token that authenticates your session. kgz uses this to:

1. **Discover** the running kernel via the Jupyter REST API
2. **Execute** code via the WebSocket Jupyter messaging protocol
3. **Stream** stdout/stderr back in real-time
4. **Transfer** files via the Jupyter Contents API

No API keys needed. No Kaggle API. Just paste the URL from your running notebook.

## API

### `Kernel`

```python
from kgz import Kernel

k = Kernel(url)              # Connect to Kaggle kernel
k.execute(code)              # Execute code (streams output)
k.run(code)                  # Alias for execute()
k.status()                   # 'idle' or 'busy'
k.interrupt()                # Stop execution (Ctrl-C)
k.wait(timeout=3600)         # Block until idle
k.is_alive()                 # True if reachable
```

### File Operations

```python
from kgz import upload_file, download_file
from kgz.file_ops import list_files, upload_directory

upload_file(url, "model.py", "model.py")
download_file(url, "/kaggle/working/results.json", "./results.json")
list_files(url)                # List files in working dir
upload_directory(url, "./src") # Upload entire directory
```

### Error Handling

```python
from kgz.kernel import KernelError

try:
    k.execute("1/0", raise_on_error=True)
except KernelError as e:
    print(e.ename)     # 'ZeroDivisionError'
    print(e.evalue)    # 'division by zero'
    print(e.traceback) # Full traceback lines
```

## Getting Your Kaggle URL

1. Open a Kaggle notebook (e.g. [kaggle.com/code](https://www.kaggle.com/code))
2. Enable GPU/TPU in Settings
3. Copy the URL from your browser — it looks like:
   ```
   https://kkb-production.jupyter-proxy.kaggle.net/k/12345/eyJhbG.../proxy
   ```
4. Pass it to kgz

The URL contains a session-specific JWT token. It expires when the notebook session ends (~12h for GPU, ~9h for TPU).

## Use Cases

- **ML Training**: Run training scripts on free Kaggle GPUs from your local editor
- **Batch Processing**: Upload data, process on Kaggle, download results
- **CI/CD**: Test GPU code on Kaggle from GitHub Actions
- **Remote Development**: Use your local IDE but compute on Kaggle

## License

MIT

## Claude Code Integration

kgz includes a `SKILL.md` for [Claude Code](https://claude.ai/claude-code). To enable it:

```bash
mkdir -p ~/.claude/skills/kgz-guide
cp SKILL.md ~/.claude/skills/kgz-guide/skill.md
```

This gives Claude Code full knowledge of the kgz API so it can use Kaggle GPUs on your behalf.

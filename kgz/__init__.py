"""
kgz — Execute code on remote Kaggle Jupyter kernels from anywhere.

Usage:
    from kgz import Kernel

    k = Kernel("https://kkb-production.jupyter-proxy.kaggle.net/k/.../proxy")
    result = k.execute("print('hello from Kaggle GPU')")
    print(result.success)    # True
    print(result.stdout)     # "hello from Kaggle GPU\n"

    k.status()               # 'idle' or 'busy'
    k.interrupt()            # Stop execution
    k.snapshot()             # Inspect remote variables
    k.resources()            # GPU/TPU usage
    k.save_session()         # Save for later
    k = Kernel.resume("name") # Resume later

CLI:
    kgz run URL "import jax; print(jax.devices())"
    kgz status URL
    kgz upload URL file.py
"""

from kgz.kernel import Kernel, CellResult, KernelError
from kgz.file_ops import upload_file, download_file, list_files, upload_directory
from kgz.sync import FileSync

__version__ = "0.1.0"
__all__ = [
    "Kernel", "CellResult", "KernelError",
    "upload_file", "download_file", "list_files", "upload_directory",
    "FileSync",
]

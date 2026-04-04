"""
kgz — Execute code on remote Kaggle Jupyter kernels (GPU & TPU).
"""
from kgz.kernel import Kernel, CellResult, KernelError
from kgz.file_ops import upload_file, download_file, list_files, upload_directory
from kgz.sync import FileSync
from kgz.cache import ResultCache
from kgz.quota import QuotaTracker
from kgz.health import KernelMonitor, parse_training_progress
from kgz.profiles import save_profile, load_profile, list_profiles
from kgz.audit import log_action, get_history

__version__ = "0.1.0"
__all__ = [
    "Kernel", "CellResult", "KernelError",
    "upload_file", "download_file", "list_files", "upload_directory",
    "FileSync", "ResultCache", "QuotaTracker",
    "KernelMonitor", "parse_training_progress",
    "save_profile", "load_profile", "list_profiles",
    "log_action", "get_history",
]

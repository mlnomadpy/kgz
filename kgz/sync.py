"""
File synchronization between local machine and Kaggle kernel.

Bidirectional sync with change detection.
"""

import os
import time
import hashlib
import threading

from kgz.file_ops import upload_file, download_file, list_files


def _file_hash(path: str) -> str:
    """Quick hash of file contents."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


class FileSync:
    """
    Watch a local directory and auto-upload changes to the kernel.

    Usage:
        sync = FileSync(url, "./src", "/kaggle/working/src")
        sync.start()    # Background thread
        # ... edit files locally ...
        sync.stop()

    Or one-shot:
        sync.push()     # Upload all changed files
        sync.pull()     # Download all remote files
    """

    def __init__(self, base_url: str, local_dir: str, remote_dir: str = "",
                 exclude: list = None):
        self.base_url = base_url
        self.local_dir = os.path.abspath(local_dir)
        self.remote_dir = remote_dir
        self.exclude = exclude or ["__pycache__", ".git", ".DS_Store", "*.pyc"]
        self._hashes: dict = {}
        self._running = False
        self._thread: object = None

    def _should_exclude(self, path: str) -> bool:
        for pattern in self.exclude:
            if pattern in path:
                return True
            if pattern.startswith("*") and path.endswith(pattern[1:]):
                return True
        return False

    def _local_files(self) -> dict:
        """Scan local directory, return {relative_path: hash}."""
        files = {}
        for root, dirs, filenames in os.walk(self.local_dir):
            dirs[:] = [d for d in dirs if not self._should_exclude(d)]
            for fname in filenames:
                full = os.path.join(root, fname)
                rel = os.path.relpath(full, self.local_dir)
                if self._should_exclude(rel):
                    continue
                files[rel] = _file_hash(full)
        return files

    def push(self) -> list:
        """Upload all changed files to remote. Returns list of uploaded paths."""
        current = self._local_files()
        uploaded = []

        for rel, h in current.items():
            if self._hashes.get(rel) != h:
                local_path = os.path.join(self.local_dir, rel)
                remote_path = f"{self.remote_dir}/{rel}" if self.remote_dir else rel
                upload_file(self.base_url, local_path, remote_path)
                uploaded.append(rel)
                self._hashes[rel] = h

        return uploaded

    def pull(self, remote_path: str = "") -> list:
        """Download all remote files to local directory."""
        path = remote_path or self.remote_dir
        files = list_files(self.base_url, path)
        downloaded = []

        for f in files:
            if f["type"] == "file":
                remote = f["path"]
                local = os.path.join(self.local_dir, f["name"])
                download_file(self.base_url, remote, local)
                downloaded.append(f["name"])

        return downloaded

    def start(self, poll_interval: float = 2.0):
        """Start watching local directory in background thread."""
        self._running = True
        self._hashes = self._local_files()  # Snapshot current state

        def _watch():
            while self._running:
                try:
                    changed = self.push()
                    if changed:
                        print(f"[kgz sync] Uploaded: {', '.join(changed)}")
                except Exception as e:
                    print(f"[kgz sync] Error: {e}")
                time.sleep(poll_interval)

        self._thread = threading.Thread(target=_watch, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop watching."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

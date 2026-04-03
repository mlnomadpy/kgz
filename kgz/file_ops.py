"""
File upload/download between local machine and Kaggle kernel via Jupyter Contents API.
"""

import os
import json
import base64
import urllib.request
from urllib.parse import urlparse


def _contents_url(base_url: str, path: str) -> str:
    """Build Jupyter Contents API URL."""
    return f"{base_url}/api/contents/{path}"


def upload_file(base_url: str, local_path: str, remote_path: str = None):
    """
    Upload a file to the Kaggle kernel's filesystem.

    Args:
        base_url: Kaggle Jupyter proxy URL
        local_path: Path to local file
        remote_path: Remote filename (default: same as local basename)
    """
    base_url = base_url.rstrip("/")
    if remote_path is None:
        remote_path = os.path.basename(local_path)

    with open(local_path, "rb") as f:
        content = f.read()

    # Try text first, fall back to base64 for binary
    try:
        text_content = content.decode("utf-8")
        payload = {
            "type": "file",
            "format": "text",
            "content": text_content,
        }
    except UnicodeDecodeError:
        payload = {
            "type": "file",
            "format": "base64",
            "content": base64.b64encode(content).decode("ascii"),
        }

    url = _contents_url(base_url, remote_path)
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method="PUT",
                                  headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req)
    result = json.loads(resp.read())
    return result.get("path", remote_path)


def download_file(base_url: str, remote_path: str, local_path: str = None):
    """
    Download a file from the Kaggle kernel's filesystem.

    Args:
        base_url: Kaggle Jupyter proxy URL
        remote_path: Remote file path
        local_path: Where to save locally (default: same basename)
    """
    base_url = base_url.rstrip("/")
    if local_path is None:
        local_path = os.path.basename(remote_path)

    url = _contents_url(base_url, remote_path)
    resp = urllib.request.urlopen(url)
    data = json.loads(resp.read())

    fmt = data.get("format", "text")
    content = data.get("content", "")

    if fmt == "base64":
        raw = base64.b64decode(content)
    else:
        raw = content.encode("utf-8") if isinstance(content, str) else content

    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    with open(local_path, "wb") as f:
        f.write(raw)

    return local_path


def list_files(base_url: str, path: str = "") -> list:
    """
    List files in a directory on the remote kernel.

    Returns list of dicts with 'name', 'path', 'type', 'size' keys.
    """
    base_url = base_url.rstrip("/")
    url = _contents_url(base_url, path)
    resp = urllib.request.urlopen(url)
    data = json.loads(resp.read())

    if data.get("type") == "directory":
        return [
            {"name": f["name"], "path": f["path"], "type": f["type"],
             "size": f.get("size", 0)}
            for f in data.get("content", [])
        ]
    return [{"name": data["name"], "path": data["path"], "type": data["type"],
             "size": data.get("size", 0)}]


def upload_directory(base_url: str, local_dir: str, remote_dir: str = ""):
    """
    Upload an entire directory tree to the kernel.
    Uses the Contents API for each file.
    """
    uploaded = []
    for root, dirs, files in os.walk(local_dir):
        for fname in files:
            if fname.startswith(".") or "__pycache__" in root:
                continue
            local_path = os.path.join(root, fname)
            rel = os.path.relpath(local_path, local_dir)
            remote_path = f"{remote_dir}/{rel}" if remote_dir else rel
            upload_file(base_url, local_path, remote_path)
            uploaded.append(remote_path)
    return uploaded

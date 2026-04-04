"""
Output caching — cache CellResults locally to avoid re-executing identical code.
"""

import os
import json
import hashlib


class ResultCache:
    """
    Cache CellResults keyed by code hash.

    Usage:
        cache = ResultCache()
        result = cache.get(code)       # None if not cached
        cache.put(code, result)        # Save result
        cache.clear()                  # Clear all
    """

    def __init__(self, cache_dir=None):
        self.cache_dir = cache_dir or os.path.expanduser("~/.kgz/cache")
        os.makedirs(self.cache_dir, exist_ok=True)

    def _key(self, code):
        return hashlib.sha256(code.encode()).hexdigest()[:16]

    def _path(self, code):
        return os.path.join(self.cache_dir, f"{self._key(code)}.json")

    def get(self, code):
        """Get cached result for code, or None."""
        path = self._path(code)
        if not os.path.exists(path):
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            from kgz.kernel import CellResult
            return CellResult(
                success=data["success"],
                stdout=data["stdout"],
                stderr=data.get("stderr", ""),
                return_value=data.get("return_value"),
                error_name=data.get("error_name"),
                error_value=data.get("error_value"),
                elapsed_seconds=data.get("elapsed_seconds", 0),
            )
        except Exception:
            return None

    def put(self, code, result):
        """Cache a result."""
        path = self._path(code)
        data = {
            "code": code[:200],  # truncate for readability
            "success": result.success,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_value": result.return_value,
            "error_name": result.error_name,
            "error_value": result.error_value,
            "elapsed_seconds": result.elapsed_seconds,
            "cached_at": __import__("time").time(),
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def has(self, code):
        return os.path.exists(self._path(code))

    def clear(self):
        """Clear all cached results."""
        import shutil
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir, exist_ok=True)

    def size(self):
        """Number of cached entries."""
        return len([f for f in os.listdir(self.cache_dir) if f.endswith(".json")])

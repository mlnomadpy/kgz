"""
Kaggle GPU/TPU quota tracking.

Kaggle limits:
- GPU: 30 hours/week
- TPU: 20 hours/week
- Sessions: 12h GPU, 9h TPU max
"""

import os
import json
import time


# Kaggle weekly quotas (hours)
WEEKLY_QUOTAS = {
    "gpu": 30.0,
    "tpu": 20.0,
}

# Max session duration (hours)
SESSION_LIMITS = {
    "gpu": 12.0,
    "tpu": 9.0,
}


class QuotaTracker:
    """
    Track Kaggle GPU/TPU usage to avoid hitting quota limits.

    Usage:
        qt = QuotaTracker()
        qt.start_session("gpu")
        # ... do work ...
        qt.end_session()
        print(qt.remaining("gpu"))  # 28.5 hours remaining
    """

    def __init__(self, path=None):
        self.path = path or os.path.expanduser("~/.kgz/quota.json")
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._data = self._load()
        self._session_start = None
        self._session_type = None

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    return json.load(f)
            except Exception:
                pass
        return {"sessions": [], "week_start": time.time()}

    def _save(self):
        with open(self.path, "w") as f:
            json.dump(self._data, f, indent=2)

    def _prune_old(self):
        """Remove sessions older than 7 days."""
        cutoff = time.time() - 7 * 24 * 3600
        self._data["sessions"] = [
            s for s in self._data["sessions"]
            if s.get("end", s.get("start", 0)) > cutoff
        ]

    def start_session(self, device_type="gpu"):
        """Start tracking a session."""
        self._session_start = time.time()
        self._session_type = device_type

    def end_session(self):
        """End current session and log usage."""
        if self._session_start is None:
            return
        elapsed = time.time() - self._session_start
        self._data["sessions"].append({
            "type": self._session_type,
            "start": self._session_start,
            "end": time.time(),
            "hours": elapsed / 3600,
        })
        self._session_start = None
        self._prune_old()
        self._save()
        return elapsed / 3600

    def used_this_week(self, device_type="gpu"):
        """Hours used this week for a device type."""
        self._prune_old()
        total = sum(
            s.get("hours", 0) for s in self._data["sessions"]
            if s.get("type") == device_type
        )
        # Add current session if running
        if self._session_start and self._session_type == device_type:
            total += (time.time() - self._session_start) / 3600
        return total

    def remaining(self, device_type="gpu"):
        """Hours remaining this week."""
        quota = WEEKLY_QUOTAS.get(device_type, 30.0)
        return max(0, quota - self.used_this_week(device_type))

    def session_time_left(self, device_type="gpu"):
        """Hours left in current session (before Kaggle auto-kills it)."""
        if self._session_start is None:
            return SESSION_LIMITS.get(device_type, 12.0)
        elapsed = (time.time() - self._session_start) / 3600
        limit = SESSION_LIMITS.get(device_type, 12.0)
        return max(0, limit - elapsed)

    def summary(self, device_type="gpu"):
        """Human-readable quota summary."""
        used = self.used_this_week(device_type)
        remaining = self.remaining(device_type)
        quota = WEEKLY_QUOTAS.get(device_type, 30.0)
        return f"{device_type.upper()}: {used:.1f}h used / {quota:.0f}h quota ({remaining:.1f}h remaining)"

    def __repr__(self):
        gpu = self.summary("gpu")
        tpu = self.summary("tpu")
        return f"QuotaTracker({gpu}, {tpu})"

"""
Audit log — records kgz actions with timestamps.
"""

import os
import json
import time
from datetime import datetime

AUDIT_PATH = os.path.expanduser("~/.kgz/audit.jsonl")


def log_action(action, kernel_name="", details=None):
    os.makedirs(os.path.dirname(AUDIT_PATH), exist_ok=True)
    entry = {"timestamp": datetime.now().isoformat(), "action": action,
             "kernel": kernel_name}
    if details: entry["details"] = details
    with open(AUDIT_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def get_history(kernel_name=None, last_n=50):
    if not os.path.exists(AUDIT_PATH): return []
    entries = []
    with open(AUDIT_PATH) as f:
        for line in f:
            if not line.strip(): continue
            try:
                e = json.loads(line)
                if kernel_name is None or e.get("kernel") == kernel_name:
                    entries.append(e)
            except: pass
    return entries[-last_n:]


def print_history(kernel_name=None, last_n=20):
    for e in get_history(kernel_name, last_n):
        ts = e.get("timestamp", "?")[:19]
        print(f"  {ts}  {e.get('kernel', ''):15s}  {e.get('action', '')}")


def clear_history():
    if os.path.exists(AUDIT_PATH): os.unlink(AUDIT_PATH)

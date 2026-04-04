"""
Kernel profiles — save and reuse Kaggle session configs.
"""

import os
import json

PROFILE_DIR = os.path.expanduser("~/.kgz/profiles")


def save_profile(name, config):
    """Save a kernel configuration profile."""
    os.makedirs(PROFILE_DIR, exist_ok=True)
    path = os.path.join(PROFILE_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    return path


def load_profile(name):
    """Load a saved profile."""
    path = os.path.join(PROFILE_DIR, f"{name}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def list_profiles():
    """List all saved profiles."""
    if not os.path.exists(PROFILE_DIR):
        return []
    return [{"profile": f[:-5], **load_profile(f[:-5])}
            for f in os.listdir(PROFILE_DIR) if f.endswith(".json")]


def delete_profile(name):
    """Delete a profile."""
    path = os.path.join(PROFILE_DIR, f"{name}.json")
    if os.path.exists(path): os.unlink(path)

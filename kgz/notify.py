"""
Notifications — Slack, webhook on Kaggle training events.
"""

import json
import urllib.request


def notify(url, message):
    """Send notification. Auto-detects Slack vs generic webhook."""
    if not url:
        return
    if "hooks.slack.com" in url:
        _slack(url, message)
    else:
        _webhook(url, {"message": message})


def _slack(webhook_url, message):
    data = json.dumps({"text": message}).encode()
    req = urllib.request.Request(webhook_url, data=data,
                                  headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        print(f"Slack failed: {e}")


def _webhook(url, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data,
                                  headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        print(f"Webhook failed: {e}")

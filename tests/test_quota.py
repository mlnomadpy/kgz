"""Tests for quota tracking."""

import os
import time
import tempfile
from kgz.quota import QuotaTracker, WEEKLY_QUOTAS, SESSION_LIMITS


class TestQuotaTracker:
    def test_initial_state(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            qt = QuotaTracker(path)
            assert qt.used_this_week("gpu") == 0.0
            assert qt.remaining("gpu") == WEEKLY_QUOTAS["gpu"]
        finally:
            os.unlink(path)

    def test_session_tracking(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            qt = QuotaTracker(path)
            qt.start_session("gpu")
            time.sleep(0.1)
            hours = qt.end_session()
            assert hours > 0
            assert qt.used_this_week("gpu") > 0
        finally:
            os.unlink(path)

    def test_remaining(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            qt = QuotaTracker(path)
            assert qt.remaining("gpu") == 30.0
            assert qt.remaining("tpu") == 20.0
        finally:
            os.unlink(path)

    def test_summary(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            qt = QuotaTracker(path)
            s = qt.summary("gpu")
            assert "GPU" in s
            assert "30" in s
        finally:
            os.unlink(path)

    def test_session_limit(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            qt = QuotaTracker(path)
            assert qt.session_time_left("gpu") == SESSION_LIMITS["gpu"]
            assert qt.session_time_left("tpu") == SESSION_LIMITS["tpu"]
        finally:
            os.unlink(path)

    def test_repr(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            qt = QuotaTracker(path)
            r = repr(qt)
            assert "GPU" in r
            assert "TPU" in r
        finally:
            os.unlink(path)

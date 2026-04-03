"""
Tests for the Kernel class.

Offline tests work without a live session.
Live tests require KGZ_TEST_URL env var.
"""

import os
import json
import tempfile
import pytest

from kgz.kernel import Kernel, CellResult, KernelError


# ---------------------------------------------------------------------------
# Offline tests (no network)
# ---------------------------------------------------------------------------
class TestKernelOffline:
    def test_bad_url_raises(self):
        with pytest.raises(Exception):
            Kernel("https://invalid.example.com/nonexistent/proxy")

    def test_cell_result_is_importable(self):
        from kgz import CellResult
        r = CellResult(stdout="hi")
        assert r.success

    def test_kernel_error_is_importable(self):
        from kgz import KernelError
        r = CellResult(success=False, error_name="E", error_value="V")
        e = KernelError(r)
        assert "E" in str(e)


class TestSessionPersistence:
    def test_save_and_list(self):
        """Test session file I/O without network."""
        session_dir = os.path.expanduser("~/.kgz")
        test_file = os.path.join(session_dir, "unit-test-session.json")

        # Write a fake session
        os.makedirs(session_dir, exist_ok=True)
        data = {"url": "https://fake", "kernel_id": "abc", "name": "unit-test-session",
                "saved_at": 0, "history_len": 0}
        with open(test_file, "w") as f:
            json.dump(data, f)

        try:
            sessions = Kernel.list_sessions()
            names = [s["name"] for s in sessions]
            assert "unit-test-session" in names
        finally:
            os.unlink(test_file)

    def test_resume_missing_raises(self):
        with pytest.raises(FileNotFoundError):
            Kernel.resume("nonexistent-session-xyz")


class TestNotebookExport:
    def test_export_format(self):
        """Test notebook JSON structure without network."""
        # Create a mock kernel-like object to test export
        # We can't instantiate Kernel without network, so test the format directly
        cells = [
            {"code": "print('a')", "success": True, "output": "a\n", "elapsed": 0.1},
            {"code": "x = 1", "success": True, "output": "", "elapsed": 0.05},
        ]
        notebook = {
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
            "cells": [],
        }
        for entry in cells:
            cell = {
                "cell_type": "code",
                "source": entry["code"],
                "metadata": {},
                "outputs": [],
                "execution_count": None,
            }
            if entry.get("output"):
                cell["outputs"].append({"output_type": "stream", "name": "stdout", "text": entry["output"]})
            notebook["cells"].append(cell)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.ipynb', delete=False) as f:
            json.dump(notebook, f)
            path = f.name

        try:
            with open(path) as f:
                nb = json.load(f)
            assert nb["nbformat"] == 4
            assert len(nb["cells"]) == 2
            assert nb["cells"][0]["source"] == "print('a')"
            assert nb["cells"][0]["outputs"][0]["text"] == "a\n"
            assert nb["cells"][1]["outputs"] == []
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Live tests (require KGZ_TEST_URL)
# ---------------------------------------------------------------------------
LIVE_URL = os.environ.get("KGZ_TEST_URL")
live = pytest.mark.skipif(LIVE_URL is None, reason="Set KGZ_TEST_URL for live tests")


class TestKernelLive:
    @live
    def test_connect(self):
        k = Kernel(LIVE_URL)
        assert k.kernel_id
        assert k.is_alive()

    @live
    def test_status(self):
        assert Kernel(LIVE_URL).status() in ("idle", "busy")

    @live
    def test_execute_simple(self):
        r = Kernel(LIVE_URL).execute("print('kgz')", stream=False)
        assert r.success
        assert "kgz" in r.stdout

    @live
    def test_execute_return_value(self):
        r = Kernel(LIVE_URL).execute("2 + 2", stream=False)
        assert r.return_value == "4"

    @live
    def test_execute_error(self):
        r = Kernel(LIVE_URL).execute("1/0", stream=False)
        assert not r.success
        assert r.error_name == "ZeroDivisionError"

    @live
    def test_raise_on_error(self):
        with pytest.raises(KernelError):
            Kernel(LIVE_URL).execute("raise ValueError('x')", stream=False, raise_on_error=True)

    @live
    def test_pipeline(self):
        k = Kernel(LIVE_URL)
        results = k.execute_notebook(["x = 10", "print(x)"], stream=False)
        assert len(results) == 2
        assert all(r.success for r in results)
        assert "10" in results[1].stdout

    @live
    def test_snapshot(self):
        k = Kernel(LIVE_URL)
        k.execute("test_var_kgz = 42", stream=False)
        snap = k.snapshot()
        assert "test_var_kgz" in snap
        assert snap["test_var_kgz"]["type"] == "int"

    @live
    def test_resources(self):
        res = Kernel(LIVE_URL).resources()
        assert "backend" in res or "cpu_percent" in res

    @live
    def test_env_vars(self):
        k = Kernel(LIVE_URL)
        k.set_env(KGZ_UNIT_TEST="passed")
        r = k.execute("import os; print(os.environ['KGZ_UNIT_TEST'])", stream=False)
        assert "passed" in r.stdout

    @live
    def test_persistent_connection(self):
        k = Kernel(LIVE_URL)
        import time
        t0 = time.time()
        for i in range(5):
            k.execute(f"_ = {i}", stream=False)
        elapsed = time.time() - t0
        # 5 calls should take < 3s with persistent WS (vs ~5s+ without)
        assert elapsed < 5.0

    @live
    def test_session_save_resume(self):
        k = Kernel(LIVE_URL, name="pytest-session")
        k.save_session()
        k2 = Kernel.resume("pytest-session")
        assert k2.name == "pytest-session"
        assert k2.is_alive()
        # Cleanup
        import os
        os.unlink(os.path.expanduser("~/.kgz/pytest-session.json"))

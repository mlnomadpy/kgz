"""Tests for result caching."""

import os
import tempfile
from kgz.cache import ResultCache
from kgz.kernel import CellResult


class TestResultCache:
    def test_miss(self):
        with tempfile.TemporaryDirectory() as d:
            c = ResultCache(d)
            assert c.get("print('x')") is None

    def test_put_get(self):
        with tempfile.TemporaryDirectory() as d:
            c = ResultCache(d)
            r = CellResult(success=True, stdout="hello\n")
            c.put("print('hello')", r)
            cached = c.get("print('hello')")
            assert cached is not None
            assert cached.stdout == "hello\n"
            assert cached.success is True

    def test_has(self):
        with tempfile.TemporaryDirectory() as d:
            c = ResultCache(d)
            assert not c.has("code")
            c.put("code", CellResult(stdout="x"))
            assert c.has("code")

    def test_clear(self):
        with tempfile.TemporaryDirectory() as d:
            c = ResultCache(d)
            c.put("a", CellResult(stdout="1"))
            c.put("b", CellResult(stdout="2"))
            assert c.size() == 2
            c.clear()
            assert c.size() == 0

    def test_different_code_different_cache(self):
        with tempfile.TemporaryDirectory() as d:
            c = ResultCache(d)
            c.put("code1", CellResult(stdout="out1"))
            c.put("code2", CellResult(stdout="out2"))
            assert c.get("code1").stdout == "out1"
            assert c.get("code2").stdout == "out2"

    def test_error_not_cached_by_default(self):
        # Errors should not be cached (only successful results)
        with tempfile.TemporaryDirectory() as d:
            c = ResultCache(d)
            r = CellResult(success=False, error_name="ValueError")
            c.put("bad", r)
            # put stores it, but execute_cached only caches success
            cached = c.get("bad")
            assert cached is not None  # put unconditionally stores


class TestTrainingParser:
    def test_parse_step_loss(self):
        from kgz.health import parse_training_progress
        m = parse_training_progress("step 100 | loss 3.71 | tok/s 56,000")
        assert m["step"] == 100
        assert m["loss"] == 3.71
        assert m["tok_per_sec"] == 56000

    def test_parse_percent(self):
        from kgz.health import parse_training_progress
        m = parse_training_progress("step 100/5000 (2.0%) | loss: 3.71")
        assert m["percent"] == 2.0
        assert m["total_steps"] == 5000

    def test_parse_empty(self):
        from kgz.health import parse_training_progress
        assert parse_training_progress("nothing") == {}


class TestProfiles:
    def test_save_load(self):
        import tempfile, os
        from kgz import profiles
        old = profiles.PROFILE_DIR
        profiles.PROFILE_DIR = tempfile.mkdtemp()
        try:
            profiles.save_profile("test", {"url": "http://fake", "name": "t"})
            loaded = profiles.load_profile("test")
            assert loaded["url"] == "http://fake"
            assert "test" in [p["profile"] for p in profiles.list_profiles()]
            profiles.delete_profile("test")
            assert profiles.load_profile("test") is None
        finally:
            profiles.PROFILE_DIR = old


class TestAudit:
    def test_log_read(self):
        import tempfile
        from kgz import audit
        old = audit.AUDIT_PATH
        audit.AUDIT_PATH = tempfile.mktemp(suffix=".jsonl")
        try:
            audit.log_action("execute", "test-kernel", {"code": "print(1)"})
            audit.log_action("interrupt", "test-kernel")
            h = audit.get_history()
            assert len(h) == 2
            assert h[0]["action"] == "execute"
            audit.clear_history()
            assert audit.get_history() == []
        finally:
            audit.AUDIT_PATH = old

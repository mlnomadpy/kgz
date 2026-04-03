"""Unit tests for FileSync — no network needed for most tests."""

import os
import tempfile

from kgz.sync import _file_hash, FileSync


class TestFileHash:
    def test_same_content_same_hash(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("hello"); p1 = f.name
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("hello"); p2 = f.name
        try:
            assert _file_hash(p1) == _file_hash(p2)
        finally:
            os.unlink(p1); os.unlink(p2)

    def test_different_content_different_hash(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("hello"); p1 = f.name
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("world"); p2 = f.name
        try:
            assert _file_hash(p1) != _file_hash(p2)
        finally:
            os.unlink(p1); os.unlink(p2)


class TestFileSyncExclude:
    def test_exclude_pycache(self):
        sync = FileSync("http://fake", "/tmp")
        assert sync._should_exclude("__pycache__/foo.pyc")
        assert sync._should_exclude("dir/__pycache__")

    def test_exclude_pyc(self):
        sync = FileSync("http://fake", "/tmp")
        assert sync._should_exclude("module.pyc")

    def test_exclude_git(self):
        sync = FileSync("http://fake", "/tmp")
        assert sync._should_exclude(".git/config")

    def test_allow_normal_files(self):
        sync = FileSync("http://fake", "/tmp")
        assert not sync._should_exclude("model.py")
        assert not sync._should_exclude("data/train.csv")

    def test_custom_exclude(self):
        sync = FileSync("http://fake", "/tmp", exclude=["*.log", "temp"])
        assert sync._should_exclude("output.log")
        assert sync._should_exclude("temp/file")
        assert not sync._should_exclude("model.py")


class TestFileSyncLocalFiles:
    def test_scans_directory(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "a.py"), "w") as f:
                f.write("x = 1")
            with open(os.path.join(d, "b.py"), "w") as f:
                f.write("y = 2")
            os.makedirs(os.path.join(d, "__pycache__"))
            with open(os.path.join(d, "__pycache__", "c.pyc"), "w") as f:
                f.write("cached")

            sync = FileSync("http://fake", d)
            files = sync._local_files()
            assert "a.py" in files
            assert "b.py" in files
            # __pycache__ excluded
            assert not any("pycache" in k for k in files)

    def test_detects_changes(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "a.py")
            with open(path, "w") as f:
                f.write("v1")

            sync = FileSync("http://fake", d)
            h1 = sync._local_files()

            with open(path, "w") as f:
                f.write("v2")

            h2 = sync._local_files()
            assert h1["a.py"] != h2["a.py"]

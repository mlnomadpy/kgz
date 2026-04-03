"""
Tests for file operations.

Offline tests use mocking. Live tests require KGZ_TEST_URL.
"""

import os
import json
import tempfile
import pytest

from kgz.file_ops import _contents_url


class TestContentsUrl:
    def test_basic(self):
        url = _contents_url("https://example.com/proxy", "file.py")
        assert url == "https://example.com/proxy/api/contents/file.py"

    def test_nested(self):
        url = _contents_url("https://example.com/proxy", "dir/file.py")
        assert url == "https://example.com/proxy/api/contents/dir/file.py"

    def test_empty_path(self):
        url = _contents_url("https://example.com/proxy", "")
        assert url == "https://example.com/proxy/api/contents/"


# Live tests
LIVE_URL = os.environ.get("KGZ_TEST_URL")
live = pytest.mark.skipif(LIVE_URL is None, reason="Set KGZ_TEST_URL for live tests")


class TestLiveFileOps:
    @live
    def test_upload_download_roundtrip(self):
        from kgz.file_ops import upload_file, download_file

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("kgz roundtrip test")
            local = f.name

        try:
            upload_file(LIVE_URL, local, "kgz_test_rt.txt")
            dl = local + ".dl"
            download_file(LIVE_URL, "kgz_test_rt.txt", dl)
            with open(dl) as f:
                assert f.read() == "kgz roundtrip test"
            os.unlink(dl)
        finally:
            os.unlink(local)

    @live
    def test_list_files(self):
        from kgz.file_ops import list_files
        files = list_files(LIVE_URL)
        assert isinstance(files, list)

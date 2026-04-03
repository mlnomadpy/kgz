"""Tests for the CLI — offline only."""

import subprocess
import sys


class TestCLI:
    def test_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "kgz.cli", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "kgz" in result.stdout.lower()

    def test_run_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "kgz.cli", "run", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "url" in result.stdout.lower()

    def test_status_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "kgz.cli", "status", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0

    def test_upload_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "kgz.cli", "upload", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0

    def test_sessions_no_args(self):
        result = subprocess.run(
            [sys.executable, "-m", "kgz.cli", "sessions"],
            capture_output=True, text=True,
        )
        # Should work without any URL
        assert result.returncode == 0

    def test_unknown_command(self):
        result = subprocess.run(
            [sys.executable, "-m", "kgz.cli", "nonexistent"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0

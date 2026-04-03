"""Unit tests for CellResult — no network needed."""

from kgz.kernel import CellResult, KernelError


class TestCellResult:
    def test_defaults(self):
        r = CellResult()
        assert r.success is True
        assert r.stdout == ""
        assert r.stderr == ""
        assert r.return_value is None
        assert r.error_name is None
        assert r.traceback == []
        assert r.display_data == []
        assert r.elapsed_seconds == 0.0

    def test_output_stdout_only(self):
        r = CellResult(stdout="hello\n")
        assert r.output == "hello\n"

    def test_output_return_value_only(self):
        r = CellResult(return_value="42")
        assert r.output == "42"

    def test_output_combined(self):
        r = CellResult(stdout="printed\n", return_value="returned")
        assert r.output == "printed\nreturned"

    def test_output_empty(self):
        r = CellResult()
        assert r.output == ""

    def test_success_repr(self):
        r = CellResult(stdout="hello")
        assert "OK" in repr(r)
        assert "hello" in repr(r)

    def test_error_repr(self):
        r = CellResult(success=False, error_name="ValueError")
        assert "ERROR" in repr(r)
        assert "ValueError" in repr(r)

    def test_long_output_truncated_in_repr(self):
        r = CellResult(stdout="x" * 100)
        assert "..." in repr(r)
        assert len(repr(r)) < 200

    def test_error_fields(self):
        r = CellResult(
            success=False,
            error_name="ZeroDivisionError",
            error_value="division by zero",
            traceback=["line 1", "line 2"],
        )
        assert not r.success
        assert r.error_name == "ZeroDivisionError"
        assert r.error_value == "division by zero"
        assert len(r.traceback) == 2

    def test_elapsed(self):
        r = CellResult(elapsed_seconds=1.5)
        assert r.elapsed_seconds == 1.5

    def test_display_data(self):
        r = CellResult(display_data=[{"text/plain": "fig", "image/png": "base64..."}])
        assert len(r.display_data) == 1
        assert "image/png" in r.display_data[0]


class TestKernelError:
    def test_from_result(self):
        r = CellResult(success=False, error_name="TypeError", error_value="bad type")
        err = KernelError(r)
        assert err.result is r
        assert "TypeError" in str(err)
        assert "bad type" in str(err)

    def test_raises(self):
        r = CellResult(success=False, error_name="RuntimeError", error_value="boom")
        try:
            raise KernelError(r)
        except KernelError as e:
            assert e.result.error_name == "RuntimeError"

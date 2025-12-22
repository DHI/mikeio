"""Tests for deprecation utilities."""

from __future__ import annotations

import warnings

import pytest

from mikeio._deprecation import _deprecate_positional_args


def test_deprecate_with_start_after() -> None:
    """Test decorator with start_after parameter (no * in signature yet)."""

    @_deprecate_positional_args(start_after="outfile")
    def scale(infile, outfile, offset=0.0, factor=1.0, items=None):
        return (infile, outfile, offset, factor, items)

    # Using keywords - no warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = scale("in.dfs", "out.dfs", offset=5.0, factor=2.0)
        assert result == ("in.dfs", "out.dfs", 5.0, 2.0, None)

    # Positional offset - should warn
    with pytest.warns(FutureWarning, match="Passing offset=5.0 as positional"):
        result = scale("in.dfs", "out.dfs", 5.0)
        assert result == ("in.dfs", "out.dfs", 5.0, 1.0, None)

    # Positional offset and factor - should warn about both
    with pytest.warns(FutureWarning, match=r"Passing offset=5.0, factor=2.0"):
        result = scale("in.dfs", "out.dfs", 5.0, 2.0)
        assert result == ("in.dfs", "out.dfs", 5.0, 2.0, None)


def test_deprecate_with_method() -> None:
    """Test that decorator works with class methods."""

    class MyClass:
        @_deprecate_positional_args(start_after="a")
        def method(self, a, b=1, c=2):
            return a + b + c

    obj = MyClass()

    # Should work without warning when using keywords
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = obj.method(1, b=2, c=3)
        assert result == 6

    # Should warn when passing keyword-only args positionally
    with pytest.warns(FutureWarning, match="Passing b=2 as positional argument"):
        result = obj.method(1, 2)
        assert result == 5


def test_deprecate_no_keyword_only_params() -> None:
    """Test that decorator does nothing when no params marked for deprecation."""

    @_deprecate_positional_args(start_after="c")
    def func(a, b, c):
        return a + b + c

    # Should work without warning (no params after c)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = func(1, 2, 3)
        assert result == 6


def test_preserves_function_metadata() -> None:
    """Test that decorator preserves function name and docstring."""

    @_deprecate_positional_args(start_after="a")
    def my_func(a, b=1):
        """This is my function."""
        return a + b

    assert my_func.__name__ == "my_func"
    assert my_func.__doc__ == "This is my function."


def test_binary_operation_signature() -> None:
    """Test with signature similar to generic.add() or generic.diff()."""

    @_deprecate_positional_args(start_after="infilename_b")
    def add(infilename_a, infilename_b, outfilename):
        return (infilename_a, infilename_b, outfilename)

    # Using keyword - no warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = add("a.dfs", "b.dfs", outfilename="c.dfs")
        assert result == ("a.dfs", "b.dfs", "c.dfs")

    # Positional outfilename - should warn
    with pytest.warns(FutureWarning, match="Passing outfilename='c.dfs'"):
        result = add("a.dfs", "b.dfs", "c.dfs")
        assert result == ("a.dfs", "b.dfs", "c.dfs")


def test_concat_signature() -> None:
    """Test with signature similar to generic.concat()."""

    @_deprecate_positional_args(start_after="outfilename")
    def concat(infilenames, outfilename, keep="last"):
        return (infilenames, outfilename, keep)

    # Using keyword - no warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = concat(["a.dfs", "b.dfs"], "out.dfs", keep="first")
        assert result == (["a.dfs", "b.dfs"], "out.dfs", "first")

    # Positional keep - should warn
    with pytest.warns(FutureWarning, match="Passing keep='first'"):
        result = concat(["a.dfs", "b.dfs"], "out.dfs", "first")
        assert result == (["a.dfs", "b.dfs"], "out.dfs", "first")


def test_extract_signature() -> None:
    """Test with signature similar to generic.extract()."""

    @_deprecate_positional_args(start_after="outfilename")
    def extract(infilename, outfilename, start=0, end=-1, step=1, items=None):
        return (infilename, outfilename, start, end, step, items)

    # Using keywords - no warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = extract("in.dfs", "out.dfs", start=10, end=20)
        assert result == ("in.dfs", "out.dfs", 10, 20, 1, None)

    # Positional start - should warn
    with pytest.warns(FutureWarning, match="Passing start=10"):
        result = extract("in.dfs", "out.dfs", 10)
        assert result == ("in.dfs", "out.dfs", 10, -1, 1, None)

    # Multiple positional - should warn about all
    with pytest.warns(FutureWarning, match=r"Passing start=10, end=20, step=2"):
        result = extract("in.dfs", "out.dfs", 10, 20, 2)
        assert result == ("in.dfs", "out.dfs", 10, 20, 2, None)


def test_warning_message_format() -> None:
    """Test that the warning message has the correct format and version info."""

    @_deprecate_positional_args(start_after="a")
    def func(a, b=1):
        return a + b

    with pytest.warns(FutureWarning) as record:
        func(1, 2)

    warning_message = str(record[0].message)
    assert "deprecated since version 3.1" in warning_message
    assert "will raise an error in version 4.0" in warning_message
    assert "Please use keyword argument(s)" in warning_message
    assert "b=2" in warning_message


def test_invalid_start_after_parameter() -> None:
    """Test that decorator raises error if start_after param doesn't exist."""

    with pytest.raises(ValueError, match="Parameter 'nonexistent' not found"):
        @_deprecate_positional_args(start_after="nonexistent")
        def func(a, b):
            return a + b

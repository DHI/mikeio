from collections.abc import Iterator

import pytest

import mikeio


@pytest.fixture(autouse=True)
def _restore_options() -> Iterator[None]:
    old = mikeio.get_options()
    yield
    mikeio.set_options(**old)


def test_default_options() -> None:
    assert mikeio.get_options()["display_max_items"] == 10


def test_get_options_returns_copy() -> None:
    opts = mikeio.get_options()
    opts["display_max_items"] = 999
    assert mikeio.get_options()["display_max_items"] == 10


def test_set_options_persists() -> None:
    mikeio.set_options(display_max_items=3)
    assert mikeio.get_options()["display_max_items"] == 3


def test_set_options_as_context_manager_restores() -> None:
    with mikeio.set_options(display_max_items=3):
        assert mikeio.get_options()["display_max_items"] == 3
    assert mikeio.get_options()["display_max_items"] == 10


def test_set_options_restores_after_exception() -> None:
    with pytest.raises(ValueError, match="boom"):
        with mikeio.set_options(display_max_items=3):
            raise ValueError("boom")
    assert mikeio.get_options()["display_max_items"] == 10


def test_unknown_option_raises() -> None:
    with pytest.raises(TypeError, match="display_max_rows"):
        mikeio.set_options(display_max_rows=3)  # type: ignore[call-arg]


@pytest.mark.parametrize("value", [-1, 1.5, "10", True])
def test_invalid_value_raises(value: object) -> None:
    with pytest.raises(ValueError, match="display_max_items"):
        mikeio.set_options(display_max_items=value)  # type: ignore[arg-type]


def test_invalid_value_leaves_option_unchanged() -> None:
    with pytest.raises(ValueError):
        mikeio.set_options(display_max_items=-1)
    assert mikeio.get_options()["display_max_items"] == 10


def test_repr_truncates_by_default() -> None:
    ds = mikeio.read("tests/testdata/sw_points.dfs0")
    lines = repr(ds).splitlines()
    assert sum(line.startswith("  0:") for line in lines) == 1
    assert "  9:" in repr(ds)
    assert "  10:" not in repr(ds)
    assert "  ... and 50 more items (60 total)" in lines


def test_repr_lists_all_items_when_limit_raised() -> None:
    ds = mikeio.read("tests/testdata/sw_points.dfs0")
    with mikeio.set_options(display_max_items=60):
        text = repr(ds)
    assert "  59:  Point 42: Mean Wave Direction, S" in text
    assert "more items" not in text


def test_repr_lists_all_items_when_limit_is_none() -> None:
    ds = mikeio.read("tests/testdata/sw_points.dfs0")
    with mikeio.set_options(display_max_items=None):
        text = repr(ds)
    assert "  59:" in text
    assert "more items" not in text


def test_repr_shows_only_count_when_limit_is_zero() -> None:
    ds = mikeio.read("tests/testdata/sw_points.dfs0")
    with mikeio.set_options(display_max_items=0):
        text = repr(ds)
    assert "number of items: 60" in text
    assert not any(line.startswith("  0:") for line in text.splitlines())


def test_option_applies_to_file_handle_repr() -> None:
    dfs = mikeio.open("tests/testdata/sw_points.dfs0")
    assert "more items" in repr(dfs)
    with mikeio.set_options(display_max_items=None):
        assert "  59:" in repr(dfs)


def test_dataset_and_file_handle_agree_at_the_limit() -> None:
    """A file with exactly display_max_items items lists them in both reprs."""
    dfs = mikeio.Dfs0("tests/testdata/sw_points.dfs0")
    ds = dfs.read()
    with mikeio.set_options(display_max_items=ds.n_items):
        assert "  0:  Buoy 2: Sign. Wave Height" in repr(dfs)
        assert "  0:  Buoy 2: Sign. Wave Height" in repr(ds)
        assert "more items" not in repr(dfs)
        assert "more items" not in repr(ds)

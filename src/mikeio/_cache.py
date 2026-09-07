"""Invalidation of cached values."""

from functools import cached_property
from typing import Any


def clear_cached_properties(obj: Any) -> None:
    """Forget all cached_property values on obj, so that they are recomputed.

    Call this from a setter that changes something the cached properties are
    derived from, e.g. the node coordinates that element_coordinates is
    calculated from.
    """
    for klass in type(obj).__mro__:
        for name, attr in vars(klass).items():
            if isinstance(attr, cached_property):
                obj.__dict__.pop(name, None)

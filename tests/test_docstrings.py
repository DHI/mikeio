"""Every public class, method and property must have a docstring.

ruff's D101/D102 rules do not catch these, because all MIKE IO modules are
underscore-prefixed and therefore private to pydocstyle - even though the
classes they hold are public and end up in the API reference.
"""

import importlib
import inspect
import pkgutil
from typing import Any

import mikeio
from mikeio import generic


def public_classes() -> list[type]:
    modules = [mikeio]
    for module_info in pkgutil.walk_packages(mikeio.__path__, "mikeio."):
        modules.append(importlib.import_module(module_info.name))

    classes = []
    for module in modules:
        for name, obj in vars(module).items():
            if name.startswith("_") or not inspect.isclass(obj):
                continue
            if not obj.__module__.startswith("mikeio"):
                continue
            if obj not in classes:
                classes.append(obj)
    return classes


def undocumented_members(cls: type) -> list[str]:
    missing = []
    for name, member in vars(cls).items():
        if name.startswith("_"):
            continue
        obj: Any = member.fget if isinstance(member, property) else member
        if not (callable(obj) or isinstance(member, property)):
            continue
        if not (getattr(obj, "__doc__", None) or "").strip():
            missing.append(name)
    return missing


def test_public_classes_have_docstrings() -> None:
    undocumented = [
        f"{cls.__module__}.{cls.__qualname__}"
        for cls in public_classes()
        if not (cls.__doc__ or "").strip()
    ]
    assert undocumented == []


def test_public_methods_and_properties_have_docstrings() -> None:
    undocumented = [
        f"{cls.__module__}.{cls.__qualname__}.{name}"
        for cls in public_classes()
        for name in undocumented_members(cls)
    ]
    assert undocumented == []


def test_public_functions_have_docstrings() -> None:
    undocumented = [
        f"{module.__name__}.{name}"
        for module in (mikeio, generic)
        for name, obj in vars(module).items()
        if not name.startswith("_")
        and inspect.isfunction(obj)
        and obj.__module__.startswith("mikeio")
        and not (obj.__doc__ or "").strip()
    ]
    assert undocumented == []

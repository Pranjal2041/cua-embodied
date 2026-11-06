"""The four competition apps. Each one pairs a software backend with GUI hooks."""
from __future__ import annotations

from importlib import import_module

from .base import App

NAMES = ("vlnverse", "dialnav", "opensun3d", "mevis")


def load_app(name: str) -> App:
    """Import one app on demand, so each app only needs its own dependencies."""
    if name not in NAMES:
        raise ValueError(f"Unknown app {name!r}; choose one of {', '.join(NAMES)}")
    return import_module(f"{__name__}.{name}").APP


__all__ = ["App", "NAMES", "load_app"]

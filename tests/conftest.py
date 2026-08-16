# Copyright (c) 2025 Resemble AI
# MIT License
"""Test-collection setup.

``chatterbox``'s real ``__init__.py`` eagerly imports ``ChatterboxTTS`` /
``ChatterboxVC`` / ``ChatterboxMultilingualTTS``, which pull in heavy,
often-not-installed runtime deps (perth, s3gen, diffusers, ...) that have
nothing to do with ``T3.loss``. To keep these tests fast and dependency-light,
install a stub ``chatterbox`` package in ``sys.modules`` (pointing at the real
source tree via ``__path__``, so submodule imports like
``chatterbox.models.t3.t3`` still resolve normally) before any test imports
the real package's ``__init__.py``.
"""
import sys
import types
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"

if "chatterbox" not in sys.modules:
    _stub = types.ModuleType("chatterbox")
    _stub.__path__ = [str(_SRC / "chatterbox")]
    sys.modules["chatterbox"] = _stub

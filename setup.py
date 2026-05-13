"""Build script for the engine.c C extension.

Project metadata (name, version, dependencies, packages, etc.) lives in
``pyproject.toml``. This file exists solely so setuptools knows to compile
``orca/engine.c`` into a platform-specific shared library at wheel-build time
(which is what ``cibuildwheel`` invokes inside the GitHub Actions release job).

The resulting binary is loaded via ``ctypes``, not as a Python C extension,
so ``engine.c`` deliberately has no ``PyInit_*`` function. setuptools is happy
to compile and link it anyway; the failure only surfaces if someone does
``import orca._engine``, which we never do (we use ``ctypes.CDLL``).

``optional=True`` means that if the compile fails for any reason (e.g. the
manylinux toolchain rejects a flag, the user's environment is exotic, etc.)
the install still succeeds with the precompiled binary missing, and
``hexgame.py``'s runtime fallback compiles ``engine.c`` on first import.
"""
from __future__ import annotations

import sys

from setuptools import Extension, setup

if sys.platform == 'win32':
    _compile_args: list[str] = ['/O2']
else:
    # Note: deliberately *no* -march=native -- baking the build machine's CPU
    # into a public wheel would break portability. Users who want max perf can
    # install from sdist (which still uses -march=native in the runtime
    # compile path) or rebuild locally.
    _compile_args = ['-O3']

engine_extension = Extension(
    name='orca._engine',
    sources=['orca/engine.c'],
    extra_compile_args=_compile_args,
    optional=True,
)

setup(ext_modules=[engine_extension])

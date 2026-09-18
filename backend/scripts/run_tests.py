#!/usr/bin/env python3
"""Fallback test runner for environments without pytest installed.

`pytest -q` is the supported way to run the suite. This script implements just
enough of the pytest API (fixtures, raises, approx, monkeypatch, importorskip,
skipif) to execute the same test files unchanged, so the suite can still be
verified on a machine where pytest is unavailable.

Usage:  python scripts/run_tests.py [test_file ...]
"""

from __future__ import annotations

import importlib.util
import inspect
import re
import sys
import traceback
import types
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# minimal pytest surface
# ---------------------------------------------------------------------------
class Skipped(Exception):
    pass


class _Approx:
    def __init__(self, expected, rel=None, abs=None):
        self.expected = expected
        self.rel = rel if rel is not None else 1e-6
        self.abs = abs if abs is not None else 1e-12

    def __eq__(self, other):
        return abs(other - self.expected) <= max(self.abs, self.rel * abs(self.expected))

    def __repr__(self):
        return f"approx({self.expected})"


class _MonkeyPatch:
    def __init__(self):
        self._undo = []

    def setenv(self, key, value):
        import os

        self._undo.append((key, os.environ.get(key)))
        os.environ[key] = value

    def delenv(self, key, raising=True):
        import os

        if key in os.environ:
            self._undo.append((key, os.environ[key]))
            del os.environ[key]
        elif raising:
            raise KeyError(key)

    def undo(self):
        import os

        for key, value in reversed(self._undo):
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self._undo.clear()


class _Mark:
    @staticmethod
    def skipif(condition, reason=""):
        def decorator(func):
            func.__skipif__ = (bool(condition), reason)
            return func

        return decorator

    @staticmethod
    def parametrize(*_args, **_kwargs):  # pragma: no cover - unused here
        raise NotImplementedError("parametrize is not supported by this fallback runner")


def _fixture(*args, **kwargs):
    def decorator(func):
        func.__is_fixture__ = True
        return func

    if args and callable(args[0]):
        return decorator(args[0])
    return decorator


@contextmanager
def _raises(expected, match=None):
    try:
        yield
    except expected as exc:  # noqa: B902
        if match and not re.search(match, str(exc)):
            raise AssertionError(f"{exc!r} does not match {match!r}") from None
    else:
        raise AssertionError(f"Expected {expected.__name__} to be raised.")


def _importorskip(name, reason=None):
    try:
        return importlib.import_module(name)
    except ImportError:
        raise Skipped(reason or f"{name} is not installed") from None


def _skip(reason=""):
    raise Skipped(reason)


pytest_shim = types.ModuleType("pytest")
pytest_shim.fixture = _fixture
pytest_shim.raises = _raises
pytest_shim.approx = _Approx
pytest_shim.mark = _Mark
pytest_shim.importorskip = _importorskip
pytest_shim.skip = _skip
pytest_shim.MonkeyPatch = _MonkeyPatch
sys.modules.setdefault("pytest", pytest_shim)

import importlib  # noqa: E402  (must follow the shim registration)


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------
def _load(path: Path):
    spec = importlib.util.spec_from_file_location(f"tests_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _collect_fixtures(*modules) -> dict:
    fixtures: dict = {}
    for module in modules:
        for name, obj in vars(module).items():
            if callable(obj) and getattr(obj, "__is_fixture__", False):
                fixtures[name] = obj
    return fixtures


def _resolve(name, fixtures, cache, finalizers, tmp_root, index):
    if name in cache:
        return cache[name]
    if name == "monkeypatch":
        patcher = _MonkeyPatch()
        finalizers.append(patcher.undo)
        cache[name] = patcher
        return patcher
    if name == "tmp_path":
        path = tmp_root / f"case_{index}"
        path.mkdir(parents=True, exist_ok=True)
        cache[name] = path
        return path
    if name not in fixtures:
        raise KeyError(f"No fixture named {name!r}")

    func = fixtures[name]
    kwargs = {
        param: _resolve(param, fixtures, cache, finalizers, tmp_root, index)
        for param in inspect.signature(func).parameters
    }
    if inspect.isgeneratorfunction(func):
        generator = func(**kwargs)
        value = next(generator)
        finalizers.append(lambda g=generator: next(g, None))
    else:
        value = func(**kwargs)
    cache[name] = value
    return value


def main(argv) -> int:
    import tempfile

    test_dir = ROOT / "tests"
    files = [Path(a) for a in argv] or sorted(test_dir.glob("test_*.py"))

    conftest = _load(test_dir / "conftest.py") if (test_dir / "conftest.py").is_file() else None
    passed = failed = skipped = 0
    failures: list = []

    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        index = 0
        for file in files:
            module = _load(file)
            fixtures = _collect_fixtures(conftest, module) if conftest else _collect_fixtures(module)
            tests = [
                (n, f)
                for n, f in sorted(vars(module).items())
                if n.startswith("test_") and callable(f) and not getattr(f, "__is_fixture__", False)
            ]
            print(f"\n{file.name}")
            for name, func in tests:
                index += 1
                condition, reason = getattr(func, "__skipif__", (False, ""))
                if condition:
                    skipped += 1
                    print(f"  s {name} ({reason})")
                    continue
                cache: dict = {}
                finalizers: list = []
                try:
                    kwargs = {
                        p: _resolve(p, fixtures, cache, finalizers, tmp_root, index)
                        for p in inspect.signature(func).parameters
                    }
                    func(**kwargs)
                    passed += 1
                    print(f"  . {name}")
                except Skipped as exc:
                    skipped += 1
                    print(f"  s {name} ({exc})")
                except Exception:  # noqa: BLE001
                    failed += 1
                    failures.append((file.name, name, traceback.format_exc()))
                    print(f"  F {name}")
                finally:
                    for finalizer in reversed(finalizers):
                        try:
                            finalizer()
                        except Exception:  # noqa: BLE001, S110
                            pass

    for file_name, test_name, tb in failures:
        print(f"\n{'=' * 70}\nFAILED {file_name}::{test_name}\n{tb}")

    print(f"\n{passed} passed, {failed} failed, {skipped} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

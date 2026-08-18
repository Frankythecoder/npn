"""Inspect the pickled artifacts in `artifacts/` and `artifacts/detectors/`.

Everything here is written with `joblib.dump` (see ml/storage/artifacts.py), so
it must be read back with `joblib.load` -- plain `pickle.load` chokes on
joblib's compressed numpy framing. Unpickling also reconstructs the
`ml.detectors.*` and `ml.features.engineer.*` classes, so the repo root has to
be importable; this script puts it on sys.path itself rather than relying on
the caller's cwd.

    python artifacts/mdlviewer.py                   # every artifact
    python artifacts/mdlviewer.py --list            # just the names
    python artifacts/mdlviewer.py isolation_forest  # one detector
    python artifacts/mdlviewer.py scalers profile_store
    python artifacts/mdlviewer.py --raw gmm         # full repr, no summary
"""
from __future__ import annotations

import argparse
import sys
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np

ARTIFACT_DIR = Path(__file__).resolve().parent
REPO_ROOT = ARTIFACT_DIR.parent
DETECTOR_DIR = ARTIFACT_DIR / "detectors"

# The pickles reference ml.detectors.* and ml.features.engineer.*, which only
# import if the repo root is on the path. Running `python artifacts/mdlviewer.py`
# puts artifacts/ there instead, so prepend it explicitly.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib  # noqa: E402  (must follow the sys.path fix-up)

PREVIEW = 4       # entries shown before eliding a dict/list
MAX_DEPTH = 2     # how far to recurse into nested containers
MAX_REPR = 300    # clamp on a fallback repr
KEY_WIDTH = 22


def summarize(value: Any, depth: int = 0) -> str:
    """One-line summary of a value, eliding big arrays and containers.

    The bundle artifacts nest -- explainer_state is a dict of dicts, the
    profile store holds per-account dicts keyed by tuples -- so this recurses a
    couple of levels and then reports shape rather than contents.
    """
    if isinstance(value, np.ndarray):
        head = f"ndarray shape={value.shape} dtype={value.dtype}"
        if value.size and np.issubdtype(value.dtype, np.number):
            head += (
                f" min={value.min():.6g} max={value.max():.6g}"
                f" mean={value.mean():.6g}"
            )
        return head

    if isinstance(value, dict):
        head = f"dict[{len(value)}]"
        if not value or depth >= MAX_DEPTH:
            return head
        shown = ", ".join(
            f"{key!r}: {summarize(item, depth + 1)}"
            for key, item in islice(value.items(), PREVIEW)
        )
        more = ", ..." if len(value) > PREVIEW else ""
        return f"{head} {{{shown}{more}}}"

    if isinstance(value, (list, tuple, set)):
        head = f"{type(value).__name__}[{len(value)}]"
        if len(value) <= PREVIEW or depth >= MAX_DEPTH:
            return head if depth >= MAX_DEPTH and value else f"{head} {value!r}"
        shown = ", ".join(
            summarize(item, depth + 1) for item in islice(value, PREVIEW)
        )
        return f"{head} [{shown}, ...]"

    text = repr(value)
    return text if len(text) <= MAX_REPR else text[:MAX_REPR] + " ..."


def _entries(obj: Any) -> tuple[str, list[tuple[str, Any]]] | None:
    """Return a (heading, items) pair for whatever kind of object this is."""
    if isinstance(obj, dict):
        return "-- entries --", [(repr(k), v) for k, v in obj.items()]
    state = getattr(obj, "__dict__", None)
    if state:
        return "-- attributes --", list(state.items())
    return None


def show(path: Path, raw: bool = False) -> None:
    obj = joblib.load(path)
    label = path.relative_to(ARTIFACT_DIR).as_posix()

    print(f"\n=== {label} ===")
    if raw:
        print(repr(obj))
        return

    cls = type(obj)
    print(f"class            : {cls.__module__}.{cls.__qualname__}")
    # Detectors carry this roster metadata as class attributes; the bundle
    # artifacts do not, hence the hasattr guard.
    for attr in ("name", "view", "scaler", "live_scorable"):
        if hasattr(obj, attr):
            print(f"{attr:<17}: {getattr(obj, attr)}")

    found = _entries(obj)
    if found is None:
        print(f"value            : {summarize(obj)}")
        return

    heading, items = found
    print(heading)
    for key, value in items:
        first, *rest = summarize(value).splitlines() or [""]
        print(f"  {key:<{KEY_WIDTH}} = {first}")
        for line in rest:
            print(" " * (KEY_WIDTH + 5) + line.strip())


def collect() -> list[Path]:
    """Every pickle in the bundle -- top-level artifacts, then detectors."""
    return sorted(ARTIFACT_DIR.glob("*.pkl")) + sorted(DETECTOR_DIR.glob("*.pkl"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "names",
        nargs="*",
        help="artifact names to show (default: all). '.pkl' optional.",
    )
    parser.add_argument(
        "--list", action="store_true", help="list available artifacts and exit"
    )
    parser.add_argument(
        "--raw", action="store_true", help="print the plain repr instead"
    )
    args = parser.parse_args()

    available = collect()
    if not available:
        print(f"no .pkl files under {ARTIFACT_DIR}", file=sys.stderr)
        return 1

    if args.list:
        for path in available:
            print(path.relative_to(ARTIFACT_DIR).as_posix())
        return 0

    if args.names:
        wanted = {name.removesuffix(".pkl") for name in args.names}
        paths = [p for p in available if p.stem in wanted]
        missing = wanted - {p.stem for p in paths}
        if missing:
            print(
                f"unknown artifact(s): {', '.join(sorted(missing))}\n"
                f"available: {', '.join(p.stem for p in available)}",
                file=sys.stderr,
            )
            return 1
    else:
        paths = available

    for path in paths:
        show(path, raw=args.raw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

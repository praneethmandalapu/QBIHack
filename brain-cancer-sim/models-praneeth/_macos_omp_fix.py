"""macOS OpenMP shim for XGBoost.

XGBoost's libxgboost.dylib needs libomp.dylib at load time. On Macs without
Homebrew, scikit-learn ships its own copy in sklearn/.dylibs/. This module
points DYLD_LIBRARY_PATH at that copy and re-execs the interpreter once, so
`import xgboost` works.

Import this at the very top of any script that uses xgboost, BEFORE importing
xgboost:

    import _macos_omp_fix  # noqa: F401  (must come first)
    import xgboost as xgb

NOTE: the re-exec breaks stdin invocation (`python - <<EOF`). Always run from a
real file: `python train_xgboost.py`.
"""

from __future__ import annotations

import os
import pathlib
import sys


def ensure_omp() -> None:
    if sys.platform != "darwin":
        return
    if os.environ.get("_OMP_FIX_DONE") == "1":
        return

    try:
        import sklearn
    except ImportError:
        return

    dylib_dir = pathlib.Path(sklearn.__file__).parent / ".dylibs"
    if not (dylib_dir / "libomp.dylib").exists():
        return

    existing = os.environ.get("DYLD_LIBRARY_PATH", "")
    os.environ["DYLD_LIBRARY_PATH"] = (
        f"{dylib_dir}:{existing}" if existing else str(dylib_dir)
    )
    os.environ["_OMP_FIX_DONE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])


ensure_omp()

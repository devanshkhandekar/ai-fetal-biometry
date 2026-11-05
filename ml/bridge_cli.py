#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

MODEL_REPO_PATH = os.environ.get("MODEL_REPO_PATH", "").strip()
if MODEL_REPO_PATH and MODEL_REPO_PATH not in sys.path:
    sys.path.insert(0, MODEL_REPO_PATH)

from real_pipeline_adapter import analyze_scan, self_test, summarize_study  # noqa: E402


def main() -> int:
    try:
        request = json.load(sys.stdin)
    except Exception as exc:
        print(f"bridge input error: {exc}", file=sys.stderr)
        return 2

    operation = request.get("operation")
    try:
        # Any prints from the existing model pipeline must go to stderr so stdout
        # remains a single valid JSON object for the platform worker.
        with contextlib.redirect_stdout(sys.stderr):
            if operation == "scan":
                result = analyze_scan(request)
            elif operation == "aggregate":
                result = summarize_study(request)
            elif operation == "self_test":
                result = self_test()
            else:
                raise ValueError(f"Unsupported operation: {operation!r}")
        sys.stdout.write(json.dumps(result, ensure_ascii=False, allow_nan=False))
        sys.stdout.flush()
        return 0
    except Exception as exc:
        import traceback

        traceback.print_exc(file=sys.stderr)
        print(f"bridge operation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

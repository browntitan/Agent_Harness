from __future__ import annotations

import os
import warnings


def test_live_acceptance_env_notice() -> None:
    missing = []
    if os.getenv("RUN_NEXT_RUNTIME_ACCEPTANCE") != "1":
        missing.append("scenario manifest")
    if os.getenv("RUN_NEXT_RUNTIME_NOTEBOOK_ACCEPTANCE") != "1":
        missing.append("notebook smoke")
    if missing:
        warnings.warn(
            "Live next-runtime acceptance was skipped for: "
            + ", ".join(missing)
            + ". Set RUN_NEXT_RUNTIME_ACCEPTANCE=1 and/or "
            + "RUN_NEXT_RUNTIME_NOTEBOOK_ACCEPTANCE=1 in a real provider/database environment.",
            stacklevel=1,
        )
    assert True

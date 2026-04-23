from __future__ import annotations

from typing import Any, Optional

from fastapi import Header, HTTPException


def require_admin_token(
    settings: Any,
    *,
    x_admin_token: Optional[str] = Header(None, alias="X-Admin-Token"),
) -> None:
    if not bool(getattr(settings, "control_panel_enabled", False)):
        raise HTTPException(status_code=404, detail="Control panel is disabled.")

    expected = str(getattr(settings, "control_panel_admin_token", "") or "").strip()
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="Control panel admin token is not configured.",
        )

    provided = str(x_admin_token or "").strip()
    if provided != expected:
        raise HTTPException(status_code=401, detail="Invalid admin token.")

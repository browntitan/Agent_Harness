from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from typing import Any


REQUIRED_SECTION_ROUTES: dict[str, list[str]] = {
    "dashboard": ["/v1/admin/overview"],
    "architecture": ["/v1/admin/architecture", "/v1/admin/architecture/activity"],
    "config": ["/v1/admin/config/schema", "/v1/admin/config/effective"],
    "agents": ["/v1/admin/agents"],
    "prompts": ["/v1/admin/prompts"],
    "collections": ["/v1/admin/collections"],
    "skills": ["/v1/skills"],
    "operations": ["/v1/admin/operations"],
}


def _fetch_json(url: str, *, headers: dict[str, str] | None = None) -> Any:
    request = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.load(response)


def _build_sections_from_paths(paths: set[str], *, reason_prefix: str = "") -> dict[str, dict[str, Any]]:
    sections: dict[str, dict[str, Any]] = {}
    for section_name, required_routes in REQUIRED_SECTION_ROUTES.items():
        missing_routes = [route for route in required_routes if route not in paths]
        supported = len(missing_routes) == 0
        if supported:
            reason = reason_prefix
        else:
            reason = "Running backend is missing one or more required routes for this section."
        sections[section_name] = {
            "supported": supported,
            "required_routes": list(required_routes),
            "missing_routes": missing_routes,
            "reason": reason,
        }
    return sections


def _all_sections_supported(sections: dict[str, dict[str, Any]]) -> bool:
    return all(bool(section.get("supported")) for section in sections.values())


def _missing_sections(sections: dict[str, dict[str, Any]]) -> list[str]:
    return [name for name, section in sections.items() if not bool(section.get("supported"))]


def main() -> None:
    parser = argparse.ArgumentParser(description="Check backend control-panel compatibility.")
    parser.add_argument("base_url")
    parser.add_argument("--token", default="")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    summary: dict[str, Any] = {
        "base_url": base_url,
        "health_ok": False,
        "source": "openapi",
        "compatible": False,
        "capabilities_route_present": False,
        "capabilities_error": "",
        "sections": {},
        "missing_sections": [],
    }

    try:
        _fetch_json(f"{base_url}/health/live")
        summary["health_ok"] = True
    except Exception as exc:  # pragma: no cover - network failure path is integration-only
        summary["capabilities_error"] = f"Health check failed: {exc}"
        json.dump(summary, sys.stdout)
        raise SystemExit(1) from exc

    try:
        openapi = _fetch_json(f"{base_url}/openapi.json")
    except Exception as exc:  # pragma: no cover - network failure path is integration-only
        summary["capabilities_error"] = f"Unable to read openapi.json: {exc}"
        json.dump(summary, sys.stdout)
        raise SystemExit(1) from exc

    paths = set((openapi or {}).get("paths", {}))
    summary["capabilities_route_present"] = "/v1/admin/capabilities" in paths

    if summary["capabilities_route_present"] and args.token:
        try:
            capabilities = _fetch_json(
                f"{base_url}/v1/admin/capabilities",
                headers={"X-Admin-Token": args.token},
            )
            sections = capabilities.get("sections") if isinstance(capabilities, dict) else None
            if isinstance(sections, dict):
                summary["source"] = "capabilities"
                summary["compatible"] = bool(capabilities.get("compatible")) and _all_sections_supported(sections)
                summary["sections"] = sections
                summary["missing_sections"] = _missing_sections(sections)
                json.dump(summary, sys.stdout)
                raise SystemExit(0 if summary["compatible"] else 1)
            summary["capabilities_error"] = "Capabilities endpoint returned an invalid payload."
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            summary["capabilities_error"] = f"Capabilities endpoint failed with HTTP {exc.code}: {detail or exc.reason}"
        except Exception as exc:  # pragma: no cover - network failure path is integration-only
            summary["capabilities_error"] = f"Capabilities endpoint failed: {exc}"
    elif summary["capabilities_route_present"] and not args.token:
        summary["capabilities_error"] = "Capabilities endpoint exists, but no admin token was provided."
    else:
        summary["capabilities_error"] = "Capabilities endpoint is missing from openapi.json."

    sections = _build_sections_from_paths(
        paths,
        reason_prefix="Derived from openapi.json only because the compatibility endpoint is unavailable.",
    )
    summary["sections"] = sections
    summary["missing_sections"] = _missing_sections(sections)
    summary["compatible"] = False
    json.dump(summary, sys.stdout)
    raise SystemExit(1)


if __name__ == "__main__":
    main()

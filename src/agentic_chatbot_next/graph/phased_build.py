from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agentic_chatbot_next.config import load_settings
from agentic_chatbot_next.graph.community_report_recovery import (
    TEXT_MODE_PHASE_1_WORKFLOWS,
    TEXT_MODE_PHASE_2_EMBED_WORKFLOWS,
    TEXT_MODE_PHASE_2_REPORT_WORKFLOWS,
    analyze_community_report_inputs,
    build_graphrag_command_prefix,
    generate_fallback_community_reports,
    rewrite_project_workflows,
    run_graphrag_cli_phase,
)


def _log(message: str) -> None:
    print(message, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the repo-owned phased GraphRAG text build.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args(argv)

    settings = load_settings()
    root_path = Path(args.root).expanduser().resolve()
    settings_path = root_path / "settings.yaml"
    if not settings_path.exists():
        _log(f"[graph-build-error] settings file not found: {settings_path}")
        return 1

    original_settings = settings_path.read_text(encoding="utf-8")
    command_prefix = build_graphrag_command_prefix(settings)
    phase_1_action = "update" if bool(args.refresh) else "index"
    fallback_used = False
    try:
        _log("[graph-build-phase] phase_1_index")
        rewrite_project_workflows(settings_path, TEXT_MODE_PHASE_1_WORKFLOWS)
        phase_1 = run_graphrag_cli_phase(
            root_path=root_path,
            command_prefix=command_prefix,
            action=phase_1_action,
            emit_log=_log,
        )
        if phase_1.returncode != 0:
            return phase_1.returncode

        repair = analyze_community_report_inputs(root_path, dry_run=False)
        _log(f"[graph-repair-summary] {json.dumps(repair, ensure_ascii=False, sort_keys=True)}")
        if bool(repair.get("native_phase2_safe", False)):
            _log("[graph-build-phase] phase_2_reports")
            rewrite_project_workflows(settings_path, TEXT_MODE_PHASE_2_REPORT_WORKFLOWS)
            phase_2 = run_graphrag_cli_phase(
                root_path=root_path,
                command_prefix=command_prefix,
                action="index",
                emit_log=_log,
            )
            if phase_2.returncode != 0:
                fallback_used = True
                _log("[graph-fallback-used] true")
                _log("[graph-build-phase] fallback_reports")
                generate_fallback_community_reports(root_path, settings=settings, emit_log=_log)
            else:
                _log("[graph-fallback-used] false")
        else:
            fallback_used = True
            _log("[graph-fallback-used] true")
            _log("[graph-build-phase] fallback_reports")
            generate_fallback_community_reports(root_path, settings=settings, emit_log=_log)

        _log("[graph-build-phase] phase_2_embeddings")
        rewrite_project_workflows(settings_path, TEXT_MODE_PHASE_2_EMBED_WORKFLOWS)
        phase_3 = run_graphrag_cli_phase(
            root_path=root_path,
            command_prefix=command_prefix,
            action="index",
            emit_log=_log,
        )
        if phase_3.returncode != 0:
            return phase_3.returncode
        _log(f"[graph-fallback-used] {'true' if fallback_used else 'false'}")
        _log("[graph-build-phase] completed")
        return 0
    finally:
        settings_path.write_text(original_settings, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

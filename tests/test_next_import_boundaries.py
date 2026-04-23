from __future__ import annotations

import ast
from pathlib import Path


LEGACY_PREFIX = "agentic_chatbot"


def _scan_python_imports(root: Path) -> list[tuple[Path, ast.AST]]:
    scanned: list[tuple[Path, ast.AST]] = []
    for path in sorted(root.rglob("*.py")):
        scanned.append((path, ast.parse(path.read_text(encoding="utf-8"), filename=str(path))))
    return scanned


def test_repo_contains_no_legacy_agentic_chatbot_imports() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    violations: list[str] = []

    for root in (
        repo_root / "src",
        repo_root / "tests",
        repo_root / "examples",
        repo_root / "new_demo_notebook" / "lib",
    ):
        for path, module in _scan_python_imports(root):
            for node in ast.walk(module):
                if isinstance(node, ast.ImportFrom) and node.module:
                    if node.module == LEGACY_PREFIX or node.module.startswith(f"{LEGACY_PREFIX}."):
                        violations.append(f"{path}: from {node.module} import ...")
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == LEGACY_PREFIX or alias.name.startswith(f"{LEGACY_PREFIX}."):
                            violations.append(f"{path}: import {alias.name}")

    assert not violations, "Legacy imports found:\n" + "\n".join(violations)

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from new_demo_notebook.lib.client import GatewayClient
from new_demo_notebook.lib.scenario_runner import ScenarioRunner, load_scenarios, validate_agent_coverage
from new_demo_notebook.lib.server import BackendServerManager


def _notebook_execution_timeout_seconds() -> int:
    raw_timeout = os.getenv("NEXT_RUNTIME_NOTEBOOK_EXECUTION_TIMEOUT_SECONDS")
    if raw_timeout in (None, ""):
        return 900
    try:
        return int(raw_timeout)
    except ValueError:
        return 900


def _build_notebook_execution_command(*, notebook_path: Path, output_dir: Path, output_name: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        f"--ExecutePreprocessor.timeout={_notebook_execution_timeout_seconds()}",
        str(notebook_path),
        "--output",
        output_name,
        "--output-dir",
        str(output_dir),
    ]


def test_build_notebook_execution_command_uses_default_timeout(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("NEXT_RUNTIME_NOTEBOOK_EXECUTION_TIMEOUT_SECONDS", raising=False)

    command = _build_notebook_execution_command(
        notebook_path=tmp_path / "demo.ipynb",
        output_dir=tmp_path / "executed",
        output_name="demo.executed.ipynb",
    )

    assert "--ExecutePreprocessor.timeout=900" in command


def test_build_notebook_execution_command_uses_env_timeout(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("NEXT_RUNTIME_NOTEBOOK_EXECUTION_TIMEOUT_SECONDS", "5400")

    command = _build_notebook_execution_command(
        notebook_path=tmp_path / "demo.ipynb",
        output_dir=tmp_path / "executed",
        output_name="demo.executed.ipynb",
    )

    assert "--ExecutePreprocessor.timeout=5400" in command


@pytest.mark.acceptance
def test_new_demo_notebook_scenarios_run_against_live_next_runtime() -> None:
    if os.getenv("RUN_NEXT_RUNTIME_ACCEPTANCE") != "1":
        pytest.skip("Set RUN_NEXT_RUNTIME_ACCEPTANCE=1 to run the live acceptance harness.")

    repo_root = Path(__file__).resolve().parents[1]
    scenarios = load_scenarios(repo_root / "new_demo_notebook" / "scenarios" / "scenarios.json")
    validate_agent_coverage(scenarios)

    artifacts_dir = repo_root / "new_demo_notebook" / ".artifacts"
    server = BackendServerManager(repo_root=repo_root, artifacts_dir=artifacts_dir)
    base_url = server.start()
    client = GatewayClient(base_url)
    try:
        model_id = client.get_model_id()
        runner = ScenarioRunner(
            client=client,
            repo_root=repo_root,
            runtime_root=repo_root / "data" / "runtime",
            workspace_root=repo_root / "data" / "workspaces",
            model_id=model_id,
        )
        for scenario in scenarios:
            result = runner.run_scenario(scenario)
            result.require_success()
    finally:
        client.close()
        server.stop(force=True)


@pytest.mark.acceptance
def test_new_demo_notebook_executes_end_to_end() -> None:
    if os.getenv("RUN_NEXT_RUNTIME_NOTEBOOK_ACCEPTANCE") != "1":
        pytest.skip("Set RUN_NEXT_RUNTIME_NOTEBOOK_ACCEPTANCE=1 to execute the live Jupyter notebook smoke test.")

    repo_root = Path(__file__).resolve().parents[1]
    notebook_path = repo_root / "new_demo_notebook" / "agentic_system_showcase.ipynb"
    output_dir = repo_root / "new_demo_notebook" / ".artifacts" / "executed"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "agentic_system_showcase.executed.ipynb"

    completed = subprocess.run(
        _build_notebook_execution_command(
            notebook_path=notebook_path,
            output_dir=output_dir,
            output_name=output_path.name,
        ),
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, (
        "Notebook execution failed.\n\n"
        f"stdout:\n{completed.stdout}\n\n"
        f"stderr:\n{completed.stderr}"
    )
    assert output_path.exists(), f"Expected executed notebook output at {output_path}"

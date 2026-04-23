from __future__ import annotations

import json
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


def _ensure_repo_import_roots() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    for candidate in (repo_root / "src", repo_root):
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)


_ensure_repo_import_roots()

from agentic_chatbot_next.memory.file_store import FileMemoryStore
from agentic_chatbot_next.runtime.context import RuntimePaths
from new_demo_notebook.lib.client import GatewayClient
from new_demo_notebook.lib.trace_reader import (
    TraceBundle,
    cleanup_conversation_artifacts,
    collect_trace_bundle,
    extract_observed_agents,
    extract_observed_event_types,
    extract_observed_route,
)

REQUIRED_AGENT_COVERAGE = [
    "basic",
    "general",
    "coordinator",
    "utility",
    "data_analyst",
    "rag_worker",
    "planner",
    "finalizer",
    "verifier",
]
_TERMINAL_JOB_STATUSES = {"completed", "failed", "stopped"}


@dataclass(frozen=True)
class ScenarioTurn:
    content: str
    force_agent: Optional[bool] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw: Any) -> "ScenarioTurn":
        if isinstance(raw, str):
            return cls(content=raw)
        if isinstance(raw, dict):
            return cls(
                content=str(raw.get("content") or ""),
                force_agent=raw.get("force_agent"),
                metadata=dict(raw.get("metadata") or {}),
            )
        raise TypeError(f"Unsupported scenario turn: {raw!r}")


@dataclass(frozen=True)
class ScenarioDefinition:
    id: str
    title: str
    description: str
    conversation_id: str
    ingest_paths: List[str]
    messages: List[ScenarioTurn]
    force_agent: bool
    expected_route: str
    expected_agents: List[str]
    expected_event_types: List[str]
    fallback_prompts: List[str]
    trace_focus: List[str]

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "ScenarioDefinition":
        return cls(
            id=str(raw.get("id") or ""),
            title=str(raw.get("title") or ""),
            description=str(raw.get("description") or ""),
            conversation_id=str(raw.get("conversation_id") or ""),
            ingest_paths=[str(item) for item in (raw.get("ingest_paths") or []) if str(item)],
            messages=[ScenarioTurn.from_raw(item) for item in (raw.get("messages") or [])],
            force_agent=bool(raw.get("force_agent", False)),
            expected_route=str(raw.get("expected_route") or ""),
            expected_agents=[str(item) for item in (raw.get("expected_agents") or []) if str(item)],
            expected_event_types=[str(item) for item in (raw.get("expected_event_types") or []) if str(item)],
            fallback_prompts=[str(item) for item in (raw.get("fallback_prompts") or []) if str(item)],
            trace_focus=[str(item) for item in (raw.get("trace_focus") or []) if str(item)],
        )


@dataclass
class ScenarioAttempt:
    attempt_index: int
    final_prompt: str
    outputs: List[str]
    raw_responses: List[Dict[str, Any]]
    observed_route: str
    observed_agents: List[str]
    observed_event_types: List[str]
    validation_errors: List[str]
    success: bool


@dataclass
class ScenarioResult:
    scenario: ScenarioDefinition
    attempts: List[ScenarioAttempt]
    history: List[Dict[str, Any]]
    bundle: TraceBundle

    @property
    def success(self) -> bool:
        return bool(self.attempts and self.attempts[-1].success)

    def require_success(self) -> None:
        if self.success:
            return
        if not self.attempts:
            raise AssertionError(f"Scenario {self.scenario.id} produced no attempts.")
        latest = self.attempts[-1]
        detail = "\n".join(f"- {item}" for item in latest.validation_errors) or "- unknown validation failure"
        raise AssertionError(
            f"Scenario {self.scenario.id} failed after {len(self.attempts)} attempt(s).\n"
            f"Observed route: {latest.observed_route}\n"
            f"Observed agents: {', '.join(latest.observed_agents)}\n"
            f"Validation errors:\n{detail}"
        )


def load_scenarios(path: Path) -> List[ScenarioDefinition]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [ScenarioDefinition.from_dict(item) for item in (payload.get("scenarios") or []) if isinstance(item, dict)]


def validate_agent_coverage(
    scenarios: List[ScenarioDefinition],
    required_agents: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    coverage: Dict[str, List[str]] = {}
    for scenario in scenarios:
        for agent in scenario.expected_agents:
            coverage.setdefault(agent, []).append(scenario.id)
    missing = [agent for agent in (required_agents or REQUIRED_AGENT_COVERAGE) if agent not in coverage]
    if missing:
        raise ValueError(f"Scenario coverage is missing agents: {', '.join(missing)}")
    return coverage


class ScenarioRunner:
    def __init__(
        self,
        *,
        client: GatewayClient,
        repo_root: Path,
        runtime_root: Path,
        workspace_root: Path,
        memory_root: Optional[Path] = None,
        model_id: str,
        job_wait_timeout_seconds: Optional[float] = None,
        trace_loader: Callable[[Path, Path, str], TraceBundle] = collect_trace_bundle,
        cleanup_fn: Callable[[Path, Path, str], None] = cleanup_conversation_artifacts,
    ) -> None:
        self.client = client
        self.repo_root = repo_root
        self.runtime_root = runtime_root
        self.workspace_root = workspace_root
        self.memory_root = memory_root or (repo_root / "data" / "memory")
        self.model_id = model_id
        self.job_wait_timeout_seconds = job_wait_timeout_seconds
        self.trace_loader = trace_loader
        self.cleanup_fn = cleanup_fn

    def run_scenario(self, scenario: ScenarioDefinition) -> ScenarioResult:
        attempts: List[ScenarioAttempt] = []
        history: List[Dict[str, Any]] = []
        bundle = TraceBundle(conversation_id=scenario.conversation_id)
        prompt_overrides = [None] + list(scenario.fallback_prompts)

        for attempt_index, override in enumerate(prompt_overrides, start=1):
            self.cleanup_fn(self.runtime_root, self.workspace_root, scenario.conversation_id)
            self._cleanup_memory_artifacts(scenario.conversation_id)
            history = []
            self._ingest_paths(scenario)
            turns = self._materialize_turns(scenario, override)
            outputs: List[str] = []
            raw_responses: List[Dict[str, Any]] = []
            for turn in turns:
                effective_force_agent = scenario.force_agent if turn.force_agent is None else bool(turn.force_agent)
                response = self.client.chat_turn(
                    history=history,
                    user_text=turn.content,
                    conversation_id=scenario.conversation_id,
                    model=self.model_id,
                    force_agent=effective_force_agent,
                    metadata=turn.metadata,
                )
                history.append({"role": "user", "content": turn.content})
                history.append({"role": "assistant", "content": response.text})
                outputs.append(response.text)
                raw_responses.append(dict(getattr(response, "raw", {}) or {}))
                self.wait_for_terminal_jobs(scenario.conversation_id)

            bundle = self.trace_loader(self.runtime_root, self.workspace_root, scenario.conversation_id)
            observed_agents = extract_observed_agents(bundle)
            observed_event_types = extract_observed_event_types(bundle)
            observed_route = extract_observed_route(bundle)
            validation_errors = self._validation_errors(
                scenario,
                observed_route=observed_route,
                observed_agents=observed_agents,
                observed_event_types=observed_event_types,
                outputs=outputs,
                raw_responses=raw_responses,
                bundle=bundle,
            )
            success = not validation_errors
            attempts.append(
                ScenarioAttempt(
                    attempt_index=attempt_index,
                    final_prompt=turns[-1].content,
                    outputs=outputs,
                    raw_responses=raw_responses,
                    observed_route=observed_route,
                    observed_agents=observed_agents,
                    observed_event_types=observed_event_types,
                    validation_errors=validation_errors,
                    success=success,
                )
            )
            if success:
                break

        return ScenarioResult(
            scenario=scenario,
            attempts=attempts,
            history=history,
            bundle=bundle,
        )

    def wait_for_terminal_jobs(self, conversation_id: str, *, timeout_seconds: Optional[float] = None) -> None:
        default_timeout = self.job_wait_timeout_seconds if self.job_wait_timeout_seconds is not None else 900.0
        effective_timeout = float(
            os.getenv(
                "NEXT_RUNTIME_JOB_WAIT_SECONDS",
                str(default_timeout if timeout_seconds is None else timeout_seconds),
            )
        )
        deadline = time.monotonic() + effective_timeout
        while time.monotonic() < deadline:
            bundle = self.trace_loader(self.runtime_root, self.workspace_root, conversation_id)
            if not bundle.jobs or all(str(job.get("status") or "") in _TERMINAL_JOB_STATUSES for job in bundle.jobs):
                return
            time.sleep(0.25)

    def _ingest_paths(self, scenario: ScenarioDefinition) -> None:
        if not scenario.ingest_paths:
            return
        resolved = [str((self.repo_root / path).resolve()) for path in scenario.ingest_paths]
        self.client.ingest(paths=resolved, conversation_id=scenario.conversation_id)

    def _materialize_turns(self, scenario: ScenarioDefinition, override_prompt: Optional[str]) -> List[ScenarioTurn]:
        turns = list(scenario.messages)
        if override_prompt is None or not turns:
            return turns
        last = turns[-1]
        turns[-1] = ScenarioTurn(content=override_prompt, force_agent=last.force_agent, metadata=dict(last.metadata))
        return turns

    def _validation_errors(
        self,
        scenario: ScenarioDefinition,
        *,
        observed_route: str,
        observed_agents: List[str],
        observed_event_types: List[str],
        outputs: List[str],
        raw_responses: List[Dict[str, Any]],
        bundle: TraceBundle,
    ) -> List[str]:
        errors: List[str] = []
        if scenario.expected_route and observed_route != scenario.expected_route:
            errors.append(f"expected route {scenario.expected_route!r}, observed {observed_route!r}")
        if scenario.expected_agents and not set(scenario.expected_agents).issubset(set(observed_agents)):
            errors.append(
                f"expected agents {sorted(set(scenario.expected_agents))}, observed {sorted(set(observed_agents))}"
            )
        if scenario.expected_event_types and not set(scenario.expected_event_types).issubset(set(observed_event_types)):
            errors.append(
                f"expected event types missing from trace: {sorted(set(scenario.expected_event_types) - set(observed_event_types))}"
            )
        if not bundle.session_ids:
            errors.append("no session runtime artifacts were captured")
        if not outputs:
            errors.append("scenario produced no assistant output")
        errors.extend(
            self._scenario_specific_errors(
                scenario,
                outputs=outputs,
                raw_responses=raw_responses,
                bundle=bundle,
            )
        )
        return errors

    def _scenario_specific_errors(
        self,
        scenario: ScenarioDefinition,
        *,
        outputs: List[str],
        raw_responses: List[Dict[str, Any]],
        bundle: TraceBundle,
    ) -> List[str]:
        scenario_id = scenario.id
        latest_output = outputs[-1] if outputs else ""
        errors: List[str] = []

        if scenario_id == "basic_route_smalltalk":
            if bundle.jobs:
                errors.append("basic route unexpectedly spawned worker jobs")
            return errors

        if scenario_id == "general_grounded_rag":
            tool_contracts = [
                payload
                for payload in self._extract_tool_payloads(bundle)
                if self._looks_like_rag_contract(payload)
            ]
            if not self._output_has_citations(latest_output, tool_contracts):
                errors.append("final output did not include a rendered citations section")
            if not tool_contracts:
                errors.append("no RAG contract-shaped tool payload was captured in persisted session state")
            return errors

        if scenario_id == "direct_grounded_rag":
            event_contracts = self._extract_event_rag_contracts(bundle)
            if not self._output_has_citations(latest_output, event_contracts):
                errors.append("final output did not include a rendered citations section")
            if not event_contracts:
                errors.append("no direct rag_contract payload was captured in persisted runtime events")
            return errors

        if scenario_id == "data_analyst_csv_review":
            expected_files = {Path(path).name for path in scenario.ingest_paths}
            observed_files = {
                name
                for names in bundle.workspace_files.values()
                for name in names
            }
            missing_files = sorted(expected_files - observed_files)
            if missing_files:
                errors.append(f"workspace did not contain expected uploaded files: {missing_files}")
            code_payloads = self._extract_code_execution_payloads(bundle)
            if not any(payload.get("success") is True for payload in code_payloads):
                if code_payloads:
                    latest = code_payloads[-1]
                    stderr = str(latest.get("stderr") or "").strip()
                    error = str(latest.get("error") or "").strip()
                    summary = stderr or error or json.dumps(latest, sort_keys=True)[:300]
                    errors.append(
                        f"no successful execute_code payload was captured in persisted session state; latest execute_code result: {summary}"
                    )
                else:
                    errors.append("no successful execute_code payload was captured in persisted session state")
            return errors

        if scenario_id == "coordinator_due_diligence":
            ordering = [
                "coordinator_planning_started",
                "coordinator_planning_completed",
                "coordinator_finalizer_completed",
                "coordinator_verifier_completed",
            ]
            if not self._events_appear_in_order(bundle.event_rows, ordering):
                errors.append("coordinator planning/finalizer/verifier events were not observed in order")
            if "rag_worker" not in {str(job.get("agent_name") or "") for job in bundle.jobs}:
                errors.append("no rag_worker job was captured for due-diligence coordination")
            return errors

        if scenario_id == "coordinator_mixed_workers":
            job_agents = {str(job.get("agent_name") or "") for job in bundle.jobs}
            for required in {"utility", "rag_worker", "general"}:
                if required not in job_agents:
                    errors.append(f"expected worker agent {required!r} was not captured in job state")
            non_terminal = [
                str(job.get("job_id") or "")
                for job in bundle.jobs
                if str(job.get("agent_name") or "") in {"utility", "rag_worker", "general"}
                and str(job.get("status") or "") not in _TERMINAL_JOB_STATUSES
            ]
            if non_terminal:
                errors.append(f"mixed-worker jobs did not reach a terminal state: {non_terminal}")
            return errors

        if scenario_id == "sync_long_form_general":
            latest_response = dict(raw_responses[-1] or {}) if raw_responses else {}
            response_metadata = dict(latest_response.get("metadata") or {})
            long_output = dict(response_metadata.get("long_output") or {})
            artifacts = [
                dict(item)
                for item in list(latest_response.get("artifacts") or [])
                if isinstance(item, dict)
            ]
            if not artifacts:
                errors.append("long-form response did not include any returned artifacts")
            if not long_output:
                errors.append("long-form response metadata did not include metadata.long_output")
                return errors
            output_filename = str(long_output.get("output_filename") or "")
            manifest_filename = str(long_output.get("manifest_filename") or "")
            if not output_filename:
                errors.append("metadata.long_output.output_filename was empty")
            if not manifest_filename:
                errors.append("metadata.long_output.manifest_filename was empty")
            observed_files = {
                name
                for names in bundle.workspace_files.values()
                for name in names
            }
            if not any(name.startswith("long_output_") for name in observed_files):
                errors.append("workspace did not contain any long_output_* files")
            for required_name in [output_filename, manifest_filename]:
                if required_name and required_name not in observed_files:
                    errors.append(f"workspace did not contain expected long-form file: {required_name}")
            for required_name in [output_filename, manifest_filename]:
                if not required_name:
                    continue
                path = self._find_workspace_file(bundle, required_name)
                if path is None:
                    continue
                if not path.read_text(encoding="utf-8").strip():
                    errors.append(f"long-form workspace file was empty: {required_name}")
            artifact_filenames = {str(item.get("filename") or "") for item in artifacts}
            if output_filename and artifact_filenames and output_filename not in artifact_filenames:
                errors.append(
                    f"long-form artifact filenames {sorted(artifact_filenames)} did not include {output_filename!r}"
                )
            if "I attached" not in latest_output and "artifact" not in latest_output.lower():
                errors.append("long-form assistant summary did not mention the attached draft artifact")
            return errors

        return errors

    def _bundle_has_task_notification(self, bundle: TraceBundle) -> bool:
        if bundle.notifications:
            return True
        for state in bundle.session_states:
            if state.get("pending_notifications"):
                return True
            for message in state.get("messages") or []:
                metadata = dict(message.get("metadata") or {})
                if isinstance(metadata.get("notification"), dict) and metadata["notification"]:
                    return True
        for row in bundle.transcript_rows:
            if str(row.get("kind") or "") == "notification":
                return True
            message = dict(row.get("message") or {})
            metadata = dict(message.get("metadata") or {})
            if isinstance(metadata.get("notification"), dict) and metadata["notification"]:
                return True
        return False

    def _load_memory_snapshot(self, bundle: TraceBundle, conversation_id: str) -> Dict[str, Dict[str, str]]:
        state = dict(bundle.session_states[-1] or {}) if bundle.session_states else {}
        tenant_id = str(state.get("tenant_id") or "local-dev")
        user_id = str(state.get("user_id") or "local-cli")
        paths = RuntimePaths(
            runtime_root=self.runtime_root,
            workspace_root=self.workspace_root,
            memory_root=self.memory_root,
        )
        store = FileMemoryStore(paths)
        snapshot: Dict[str, Dict[str, str]] = {"conversation": {}, "user": {}}
        for scope in ("conversation", "user"):
            entries = store.list_entries(
                tenant_id=tenant_id,
                user_id=user_id,
                conversation_id=conversation_id,
                scope=scope,
            )
            snapshot[scope] = {entry.key: entry.value for entry in entries}
        return snapshot

    def _extract_tool_payloads(self, bundle: TraceBundle) -> List[Dict[str, Any]]:
        payloads: List[Dict[str, Any]] = []
        for state in bundle.session_states:
            for message in state.get("messages") or []:
                if str(message.get("role") or "") != "tool":
                    continue
                content = str(message.get("content") or "").strip()
                if not content:
                    continue
                try:
                    parsed = json.loads(content)
                except Exception:
                    continue
                if isinstance(parsed, dict):
                    payloads.append(parsed)
        return payloads

    def _looks_like_rag_contract(self, payload: Dict[str, Any]) -> bool:
        required_keys = {"answer", "citations", "used_citation_ids", "followups", "warnings"}
        return required_keys.issubset(set(payload.keys()))

    def _extract_code_execution_payloads(self, bundle: TraceBundle) -> List[Dict[str, Any]]:
        payloads: List[Dict[str, Any]] = []
        for payload in self._extract_tool_payloads(bundle):
            keys = set(payload.keys())
            if {"stdout", "stderr", "success"}.issubset(keys) or "execution_time_seconds" in keys:
                payloads.append(payload)
        return payloads

    def _extract_event_rag_contracts(self, bundle: TraceBundle) -> List[Dict[str, Any]]:
        contracts: List[Dict[str, Any]] = []
        for row in bundle.event_rows + bundle.job_events:
            payload = dict(row.get("payload") or {})
            direct = payload.get("rag_contract")
            if isinstance(direct, dict) and self._looks_like_rag_contract(direct):
                contracts.append(direct)
            elif self._looks_like_rag_contract(payload):
                contracts.append(payload)
        for state in bundle.session_states:
            for message in state.get("messages") or []:
                metadata = dict(message.get("metadata") or {})
                direct = metadata.get("rag_contract")
                if isinstance(direct, dict) and self._looks_like_rag_contract(direct):
                    contracts.append(direct)
        return contracts

    def _extract_rag_contracts(self, bundle: TraceBundle) -> List[Dict[str, Any]]:
        contracts = [
            payload
            for payload in self._extract_tool_payloads(bundle)
            if self._looks_like_rag_contract(payload)
        ]
        contracts.extend(self._extract_event_rag_contracts(bundle))
        return contracts

    def _output_has_citations(self, output: str, contracts: List[Dict[str, Any]]) -> bool:
        text = str(output or "")
        if re.search(r"(?im)^\s{0,3}(?:#+\s*)?citations\s*:?\s*$", text):
            return True
        if re.search(r"\bKB_[A-Za-z0-9]+#chunk\d+\b", text):
            return True
        return any(isinstance(contract.get("citations"), list) and bool(contract.get("citations")) for contract in contracts)

    def _events_appear_in_order(self, rows: List[Dict[str, Any]], expected_types: List[str]) -> bool:
        indices: List[int] = []
        for expected in expected_types:
            for idx, row in enumerate(rows):
                if str(row.get("event_type") or "") == expected:
                    indices.append(idx)
                    break
            else:
                return False
        return indices == sorted(indices)

    def _find_workspace_file(self, bundle: TraceBundle, filename: str) -> Path | None:
        wanted = str(filename or "").strip()
        if not wanted:
            return None
        for root in bundle.workspace_roots.values():
            candidate = Path(root) / wanted
            if candidate.exists() and candidate.is_file():
                return candidate
        return None

    def _cleanup_memory_artifacts(self, conversation_id: str) -> None:
        paths = RuntimePaths(
            runtime_root=self.runtime_root,
            workspace_root=self.workspace_root,
            memory_root=self.memory_root,
        )
        tenant_id = os.getenv("DEFAULT_TENANT_ID", "local-dev")
        user_id = os.getenv("DEFAULT_USER_ID", "local-cli")
        shutil.rmtree(
            paths.conversation_memory_dir(tenant_id, user_id, conversation_id),
            ignore_errors=True,
        )
        shutil.rmtree(
            paths.user_profile_dir(tenant_id, user_id),
            ignore_errors=True,
        )

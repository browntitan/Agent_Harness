from __future__ import annotations

import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentic_chatbot_next.documents.evidence import EvidenceBinderService
from agentic_chatbot_next.documents.templates import TemplateTransformService
from agentic_chatbot_next.runtime.artifacts import register_handoff_artifact
from agentic_chatbot_next.sandbox.workspace import SessionWorkspace
from agentic_chatbot_next.tools.document_tools import make_document_tools
from agentic_chatbot_next.tools.registry import build_tool_definitions


def _workspace(tmp_path: Path) -> SessionWorkspace:
    workspace = SessionWorkspace.for_session("session-evidence-binder", tmp_path / "workspaces")
    workspace.open()
    return workspace


def _session(workspace: SessionWorkspace) -> SimpleNamespace:
    return SimpleNamespace(
        tenant_id="tenant",
        user_id="user",
        session_id="session-evidence-binder",
        conversation_id="conversation-evidence-binder",
        metadata={},
        workspace=workspace,
    )


def _settings(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        default_tenant_id="tenant",
        default_collection_id="default",
        docling_enabled=False,
        workspace_dir=tmp_path / "workspaces",
    )


def _stores() -> SimpleNamespace:
    return SimpleNamespace(doc_store=SimpleNamespace(), chunk_store=SimpleNamespace())


def _write_workspace_file(workspace: SessionWorkspace, tmp_path: Path, filename: str, body: str) -> None:
    source = tmp_path / filename
    source.write_text(body, encoding="utf-8")
    workspace.copy_file(source)


SOURCE_TEXT = """UNCLASSIFIED

# Sustainment Review

The contractor shall submit CDRL A001 Program Management Report monthly in PDF format.

The supplier must maintain a verification log for all system tests.

Test results showed pass for thermal cycle and one anomaly for vibration restart.
"""


def _artifact_path(workspace: SessionWorkspace, artifacts: list[dict], suffix: str) -> Path:
    filename = next(item["filename"] for item in artifacts if item["filename"].endswith(suffix))
    return workspace.root / filename


def test_evidence_binder_builds_docx_and_zip_with_sources_and_latest_artifacts(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    session = _session(workspace)
    _write_workspace_file(workspace, tmp_path, "sustainment.txt", SOURCE_TEXT)
    TemplateTransformService(_settings(tmp_path), _stores(), session).transform(
        document_refs=["sustainment.txt"],
        source_scope="workspace",
        template_type="memo",
        output_format="auto",
    )

    result = EvidenceBinderService(_settings(tmp_path), _stores(), session).build(
        binder_title="Sustainment Evidence",
        objective="Package source-backed memo evidence.",
        document_refs=["sustainment.txt"],
        source_scope="workspace",
        include_latest_artifacts=True,
    )

    assert result.status == "completed"
    assert len(result.source_documents) == 1
    assert result.evidence_rows
    assert any(item["filename"].endswith(".docx") for item in result.binder_artifacts)
    assert any(item["filename"].endswith(".zip") for item in result.binder_artifacts)
    assert any(artifact.included_in_zip for artifact in result.artifacts)

    from docx import Document

    document = Document(str(_artifact_path(workspace, result.binder_artifacts, ".docx")))
    docx_text = "\n".join(paragraph.text for paragraph in document.paragraphs)
    assert "Source Inventory" in docx_text
    assert "Claim To Evidence Table" in docx_text

    with zipfile.ZipFile(_artifact_path(workspace, result.binder_artifacts, ".zip")) as archive:
        names = set(archive.namelist())
        assert "evidence_manifest.json" in names
        assert "evidence_table.csv" in names
        assert "source_excerpts.jsonl" in names
        assert "open_issues.md" in names
        assert any(name.startswith("included_artifacts/") for name in names)
        assert "original_sources/sustainment.txt" not in names


def test_evidence_binder_resolves_explicit_workspace_artifact_filename(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    session = _session(workspace)
    (workspace.root / "loose_note.md").write_text("# Review Note\n\nGenerated summary without trace.", encoding="utf-8")

    result = EvidenceBinderService(_settings(tmp_path), _stores(), session).build(
        artifact_refs=["loose_note.md"],
        include_latest_artifacts=False,
        citation_policy="warn_and_include",
    )

    assert len(result.artifacts) == 1
    assert result.artifacts[0].artifact_ref == "workspace://loose_note.md"
    assert result.artifacts[0].included_in_zip is True
    assert result.open_issues
    with zipfile.ZipFile(_artifact_path(workspace, result.binder_artifacts, ".zip")) as archive:
        assert "included_artifacts/loose_note.md" in set(archive.namelist())


def test_evidence_binder_includes_handoff_artifacts_by_type(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    session = _session(workspace)
    handoff = register_handoff_artifact(
        session,
        artifact_type="consolidation_clusters",
        handoff_schema="consolidation_clusters.v1",
        producer_task_id="campaign-1",
        producer_agent="document_consolidation_campaign",
        summary="Two recommended consolidation clusters.",
        data={"clusters": [{"cluster_id": "cluster-1"}]},
    )

    result = EvidenceBinderService(_settings(tmp_path), _stores(), session).build(
        handoff_artifact_ids=[handoff["artifact_id"]],
        handoff_artifact_types=["consolidation_clusters"],
        include_latest_artifacts=False,
    )

    assert len(result.handoff_artifacts) == 1
    assert result.handoff_artifacts[0].artifact_type == "consolidation_clusters"
    assert result.evidence_rows[0].citation_status == "weak"
    assert result.open_issues


def test_evidence_binder_citation_policies_for_unsupported_artifacts(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    session = _session(workspace)
    (workspace.root / "unsupported.md").write_text("Generated statement without source trace.", encoding="utf-8")

    warn_result = EvidenceBinderService(_settings(tmp_path), _stores(), session).build(
        artifact_refs=["unsupported.md"],
        include_latest_artifacts=False,
        citation_policy="warn_and_include",
    )
    assert any(row.review_status == "needs_review" for row in warn_result.evidence_rows)

    exclude_result = EvidenceBinderService(_settings(tmp_path), _stores(), session).build(
        artifact_refs=["unsupported.md"],
        include_latest_artifacts=False,
        citation_policy="exclude_unsupported",
    )
    assert exclude_result.evidence_rows == []
    assert exclude_result.open_issues

    with pytest.raises(ValueError, match="citation policy failed"):
        EvidenceBinderService(_settings(tmp_path), _stores(), session).build(
            artifact_refs=["unsupported.md"],
            include_latest_artifacts=False,
            citation_policy="fail_closed",
        )


def test_evidence_binder_tool_returns_compact_payload_and_registry_metadata(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    session = _session(workspace)
    _write_workspace_file(workspace, tmp_path, "source.txt", SOURCE_TEXT)
    tools = {
        tool.name: tool
        for tool in make_document_tools(
            _settings(tmp_path),
            _stores(),
            session,
        )
    }

    result = tools["evidence_binder"].invoke(
        {
            "binder_title": "Compact Evidence",
            "document_refs": ["source.txt"],
            "source_scope": "workspace",
            "include_latest_artifacts": False,
            "run_in_background": False,
        }
    )

    assert result["status"] == "completed"
    assert result["source_document_count"] == 1
    assert "source_documents" not in result
    assert any(item["filename"].endswith(".docx") for item in result["binder_artifacts"])

    definition = build_tool_definitions(None)["evidence_binder"]
    assert definition.read_only is True
    assert definition.requires_workspace is True
    assert definition.background_safe is True
    assert definition.concurrency_key == "evidence_binder"

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from agentic_chatbot_next.documents.consolidation import DocumentConsolidationCampaignService
from agentic_chatbot_next.sandbox.workspace import SessionWorkspace
from agentic_chatbot_next.tools.document_tools import make_document_tools
from agentic_chatbot_next.tools.registry import build_tool_definitions


def _workspace(tmp_path: Path) -> SessionWorkspace:
    workspace = SessionWorkspace.for_session("session-consolidation", tmp_path / "workspaces")
    workspace.open()
    return workspace


def _session(workspace: SessionWorkspace) -> SimpleNamespace:
    return SimpleNamespace(
        tenant_id="tenant",
        user_id="user",
        session_id="session-consolidation",
        conversation_id="conversation-consolidation",
        metadata={},
        workspace=workspace,
    )


def _settings(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        default_tenant_id="tenant",
        default_collection_id="default",
        docling_enabled=False,
        workspace_dir=tmp_path / "workspaces",
        document_consolidation_boilerplate_terms="asterion systems, proprietary, internal use only",
    )


def _stores() -> SimpleNamespace:
    return SimpleNamespace(doc_store=SimpleNamespace(), chunk_store=SimpleNamespace())


def _write_workspace_doc(workspace: SessionWorkspace, tmp_path: Path, filename: str, body: str) -> None:
    source = tmp_path / filename
    source.write_text(body, encoding="utf-8")
    workspace.copy_file(source)


ONBOARDING_FLOW_A = """Asterion Systems proprietary. Internal use only.

# Supplier Onboarding Procedure

1. Intake coordinator receives the supplier request form.

2. Compliance analyst reviews export-control screening and risk flags.

3. Procurement manager approves the vendor setup ticket.

4. Records specialist archives the approval package.
"""


ONBOARDING_FLOW_B = """Asterion Systems proprietary. Internal use only.

# Vendor Activation Work Instruction

1. The intake coordinator receives the vendor intake packet.

2. The compliance analyst validates export screening and risk indicators.

3. The procurement manager approves the supplier setup ticket.

4. The records specialist archives the completed approval package.
"""


TRAVEL_POLICY = """Asterion Systems proprietary. Internal use only.

# Travel Expense Policy

Employees submit travel receipts within five business days.

Finance reviews lodging limits and reimburses approved expenses.

Managers may reject late claims that lack a receipt.
"""


CYBER_INCIDENT = """Asterion Systems proprietary. Internal use only.

# Cyber Incident Response

1. Security operations receives an incident alert from monitoring.

2. Incident commander isolates affected systems.

3. Forensics analyst preserves volatile evidence.

4. Communications lead prepares the stakeholder notification.
"""


def test_consolidation_campaign_identifies_same_sector_process_flow_candidates(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    session = _session(workspace)
    _write_workspace_doc(workspace, tmp_path, "Alpha - Supplier Onboarding.txt", ONBOARDING_FLOW_A)
    _write_workspace_doc(workspace, tmp_path, "Alpha - Vendor Activation.txt", ONBOARDING_FLOW_B)
    _write_workspace_doc(workspace, tmp_path, "Alpha - Travel Policy.txt", TRAVEL_POLICY)

    result = DocumentConsolidationCampaignService(_settings(tmp_path), _stores(), session).run(
        source_scope="workspace",
        sector_mode="title_prefix",
        similarity_focus="process_flows",
        export=True,
    )

    assert result.selected_document_count == 3
    assert result.consolidation_clusters
    top = result.consolidation_clusters[0]
    titles = {doc.title for doc in top.documents}
    assert "Alpha - Supplier Onboarding.txt" in titles
    assert "Alpha - Vendor Activation.txt" in titles
    assert "shared_process_flow" in top.reason_codes
    assert all("Travel Policy" not in doc.title for doc in top.documents)
    assert any(item["filename"] == "document_consolidation_report.md" for item in result.artifacts)


def test_consolidation_similarity_downweights_common_company_boilerplate(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    session = _session(workspace)
    _write_workspace_doc(workspace, tmp_path, "Alpha - Travel Policy.txt", TRAVEL_POLICY)
    _write_workspace_doc(workspace, tmp_path, "Alpha - Cyber Incident.txt", CYBER_INCIDENT)

    result = DocumentConsolidationCampaignService(_settings(tmp_path), _stores(), session).run(
        source_scope="workspace",
        sector_mode="title_prefix",
        similarity_focus="process_flows",
        min_similarity_score=0.72,
        export=False,
    )

    assert result.similarity_edges == []
    assert result.consolidation_clusters == []


def test_consolidation_cross_sector_modes_block_report_and_allow(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    session = _session(workspace)
    _write_workspace_doc(workspace, tmp_path, "Alpha - Supplier Onboarding.txt", ONBOARDING_FLOW_A)
    _write_workspace_doc(workspace, tmp_path, "Beta - Supplier Onboarding.txt", ONBOARDING_FLOW_B)

    service = DocumentConsolidationCampaignService(_settings(tmp_path), _stores(), session)
    blocked = service.run(
        source_scope="workspace",
        sector_mode="title_prefix",
        cross_sector_mode="blocked",
        similarity_focus="process_flows",
        export=False,
    )
    assert blocked.similarity_edges == []
    assert blocked.consolidation_clusters == []

    advisory = service.run(
        source_scope="workspace",
        sector_mode="title_prefix",
        cross_sector_mode="report_only",
        similarity_focus="process_flows",
        export=False,
    )
    assert advisory.similarity_edges
    assert advisory.similarity_edges[0].cross_sector_advisory is True
    assert advisory.consolidation_clusters == []

    allowed = service.run(
        source_scope="workspace",
        sector_mode="title_prefix",
        allow_cross_sector_comparisons=True,
        similarity_focus="process_flows",
        export=False,
    )
    assert allowed.similarity_edges
    assert allowed.consolidation_clusters
    assert allowed.consolidation_clusters[0].cross_sector is True


def test_document_consolidation_tool_returns_compact_artifacts(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    session = _session(workspace)
    _write_workspace_doc(workspace, tmp_path, "Alpha - Supplier Onboarding.txt", ONBOARDING_FLOW_A)
    _write_workspace_doc(workspace, tmp_path, "Alpha - Vendor Activation.txt", ONBOARDING_FLOW_B)

    tools = {
        tool.name: tool
        for tool in make_document_tools(
            _settings(tmp_path),
            _stores(),
            session,
        )
    }
    result = tools["document_consolidation_campaign"].invoke(
        {
            "source_scope": "workspace",
            "sector_mode": "title_prefix",
            "similarity_focus": "process_flows",
            "run_in_background": False,
            "export": True,
        }
    )

    assert result["status"] == "completed"
    assert result["candidate_cluster_count"] == 1
    assert "manifest" not in result
    filenames = {item["filename"] for item in result["artifacts"]}
    assert {
        "consolidation_manifest.json",
        "document_similarity_edges.jsonl",
        "process_flow_matches.jsonl",
        "consolidation_clusters.json",
        "sector_summary.json",
        "document_consolidation_report.md",
    }.issubset(filenames)


def test_document_consolidation_registry_metadata() -> None:
    definitions = build_tool_definitions(None)

    definition = definitions["document_consolidation_campaign"]
    assert definition.read_only is True
    assert definition.requires_workspace is True
    assert definition.background_safe is True
    assert definition.concurrency_key == "document_campaign"

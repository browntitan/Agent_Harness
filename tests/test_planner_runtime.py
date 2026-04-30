from __future__ import annotations

from types import SimpleNamespace

from agentic_chatbot_next.contracts.agents import AgentDefinition
from agentic_chatbot_next.contracts.messages import SessionState
from agentic_chatbot_next.runtime.query_loop import QueryLoop
from agentic_chatbot_next.runtime.task_plan import build_fallback_plan, normalise_task_plan, select_execution_batch
from agentic_chatbot_next.runtime.turn_contracts import resolve_turn_intent


class RecordingChatModel:
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.calls: list[dict[str, object]] = []

    def invoke(self, messages, config=None):
        self.calls.append({"messages": list(messages), "config": dict(config or {})})
        return SimpleNamespace(content=self.response_text)


def test_single_step_request_produces_one_sequential_task():
    plan = build_fallback_plan("Summarize the latest uploaded policy.")

    assert len(plan) == 1
    assert plan[0]["mode"] == "sequential"
    assert plan[0]["executor"] == "rag_worker"
    assert plan[0]["coverage_goal"] == "targeted"
    assert plan[0]["result_mode"] == "answer"


def test_comparison_request_produces_parallel_tasks():
    plan = build_fallback_plan('Compare "MSA v1" and "MSA v2" for indemnity differences.')

    assert len(plan) >= 2
    assert all(task["mode"] == "parallel" for task in plan)
    assert all(task["executor"] == "rag_worker" for task in plan)
    assert all(task["research_profile"] == "comparison_campaign" for task in plan)


def test_corpus_discovery_request_produces_multi_task_campaign():
    plan = build_fallback_plan("Identify all documents that have process flows outlined in them.")

    assert [task["id"] for task in plan] == ["task_1", "task_2", "task_3", "task_4", "task_5", "task_6", "task_7", "task_8"]
    assert plan[0]["executor"] == "general"
    assert plan[0]["produces_artifacts"] == ["title_candidates"]
    assert plan[1]["executor"] == "rag_worker"
    assert plan[1]["answer_mode"] == "evidence_only"
    assert plan[1]["consumes_artifacts"] == ["title_candidates"]
    assert plan[1]["produces_artifacts"] == ["doc_focus"]
    assert plan[2]["executor"] == "general"
    assert plan[2]["consumes_artifacts"] == ["title_candidates", "doc_focus"]
    assert plan[2]["produces_artifacts"] == ["research_facets"]
    assert plan[3]["controller_hints"]["dynamic_facet_fanout"] is True
    assert plan[4]["controller_hints"]["dynamic_triage_fanout"] is True
    assert plan[5]["controller_hints"]["dynamic_doc_review_fanout"] is True
    assert "research_triage_note" in plan[5]["consumes_artifacts"]
    assert plan[6]["produces_artifacts"] == ["subsystem_inventory"]
    assert plan[7]["controller_hints"]["dynamic_subsystem_backfill"] is True
    assert all(task["controller_hints"].get("prefer_inventory_output") for task in plan)


def test_structured_document_discovery_prompt_produces_corpus_discovery_plan():
    plan = build_fallback_plan(
        """
        Goal: Investigate the major subsystems in this repo and provide me a list of documents that have information about the major sub systems
        Context: The documents i want you to pull this information from is provided to you internally in the knowledge base we have access to in the "default" collection, conduct your search accross the documents that are in there
        Deliverable: you will provide me only a list of potential documents that have information about major subsystems
        """
    )

    assert [task["id"] for task in plan] == ["task_1", "task_2", "task_3", "task_4", "task_5", "task_6", "task_7", "task_8"]
    assert plan[0]["executor"] == "general"
    assert plan[0]["produces_artifacts"] == ["title_candidates"]
    assert plan[1]["executor"] == "rag_worker"
    assert plan[1]["answer_mode"] == "evidence_only"
    assert plan[1]["coverage_goal"] == "corpus_wide"
    assert plan[1]["result_mode"] == "inventory"
    assert plan[2]["executor"] == "general"
    assert plan[2]["produces_artifacts"] == ["research_facets"]
    assert plan[4]["controller_hints"]["dynamic_triage_fanout"] is True
    assert plan[5]["controller_hints"]["dynamic_doc_review_fanout"] is True
    assert "research_triage_note" in plan[5]["consumes_artifacts"]
    assert plan[6]["produces_artifacts"] == ["subsystem_inventory"]
    assert all("default" not in task["doc_scope"] for task in plan)
    assert plan[0]["controller_hints"]["kb_collection_id"] == "default"


def test_exact_major_subsystems_prompt_is_holistic_repository_research():
    query = "Investigate the major subsystems in this repo and give me a concise architectural walkthrough with citations."

    intent = resolve_turn_intent(query, {"kb_collection_id": "default"})

    assert intent.answer_contract.kind == "grounded_synthesis"
    assert intent.answer_contract.depth == "deep"
    assert intent.answer_contract.broad_coverage is True
    assert intent.answer_contract.coverage_profile == "holistic_repository"
    assert intent.answer_contract.final_output_mode == "detailed_subsystem_summary"


def test_exact_major_subsystems_prompt_produces_multi_stage_research_campaign():
    query = "Investigate the major subsystems in this repo and give me a concise architectural walkthrough with citations."

    plan = build_fallback_plan(query, session_metadata={"kb_collection_id": "default"})

    assert [task["id"] for task in plan] == ["task_1", "task_2", "task_3", "task_4", "task_5", "task_6", "task_7", "task_8"]
    assert plan[0]["produces_artifacts"] == ["title_candidates"]
    assert plan[1]["executor"] == "rag_worker"
    assert plan[1]["answer_mode"] == "evidence_only"
    assert plan[2]["produces_artifacts"] == ["research_facets"]
    assert plan[4]["controller_hints"]["dynamic_triage_fanout"] is True
    assert plan[5]["controller_hints"]["dynamic_doc_review_fanout"] is True
    assert "research_triage_note" in plan[5]["consumes_artifacts"]
    assert plan[6]["produces_artifacts"] == ["subsystem_inventory"]
    assert plan[7]["controller_hints"]["dynamic_subsystem_backfill"] is True
    assert all(task["controller_hints"]["final_output_mode"] == "detailed_subsystem_summary" for task in plan)


def test_active_doc_focus_followup_produces_exact_doc_summary_plan():
    plan = build_fallback_plan(
        "Can you look through the candidate documents you provided and give me a detailed summary of the major subsystems involved?",
        session_metadata={
            "active_doc_focus": {
                "collection_id": "default",
                "documents": [
                    {"doc_id": "KB_architecture", "title": "ARCHITECTURE.md"},
                    {"doc_id": "KB_control_flow", "title": "CONTROL_FLOW.md"},
                    {"doc_id": "KB_gateway", "title": "OPENAI_GATEWAY.md"},
                ],
            }
        },
    )

    assert [task["id"] for task in plan] == ["task_1", "task_2", "task_3", "task_4", "task_5"]
    assert all(task["executor"] == "general" for task in plan[:3])
    assert all(task["mode"] == "parallel" for task in plan[:3])
    assert all(task["produces_artifacts"] == ["doc_digest"] for task in plan[:3])
    assert [task["doc_scope"][0] for task in plan[:3]] == [
        "KB_architecture",
        "KB_control_flow",
        "KB_gateway",
    ]
    assert plan[3]["executor"] == "general"
    assert plan[3]["consumes_artifacts"] == ["doc_digest"]
    assert plan[3]["produces_artifacts"] == ["subsystem_inventory"]
    assert plan[4]["controller_hints"]["dynamic_subsystem_backfill"] is True
    assert all(task["controller_hints"]["summary_scope"] == "active_doc_focus" for task in plan)
    assert all(task["controller_hints"]["final_output_mode"] == "detailed_subsystem_summary" for task in plan)
    assert all(task["controller_hints"]["strict_doc_focus"] is True for task in plan)
    assert all(task["controller_hints"]["doc_read_depth"] == "full" for task in plan)


def test_active_doc_focus_followup_uses_all_candidate_docs_even_when_max_tasks_is_small():
    plan = build_fallback_plan(
        "Summarize the candidate documents you provided in detail.",
        max_tasks=4,
        session_metadata={
            "active_doc_focus": {
                "collection_id": "default",
                "documents": [
                    {"doc_id": "KB_architecture", "title": "ARCHITECTURE.md"},
                    {"doc_id": "KB_control_flow", "title": "CONTROL_FLOW.md"},
                    {"doc_id": "KB_gateway", "title": "OPENAI_GATEWAY.md"},
                    {"doc_id": "KB_composition", "title": "COMPOSITION.md"},
                ],
            }
        },
    )

    assert [task["title"] for task in plan[:4]] == [
        "Digest ARCHITECTURE.md",
        "Digest CONTROL_FLOW.md",
        "Digest OPENAI_GATEWAY.md",
        "Digest COMPOSITION.md",
    ]
    assert plan[4]["title"] == "Consolidate subsystem inventory"
    assert plan[5]["title"] == "Expand subsystem evidence backfill"


def test_exact_named_document_prompt_stays_focused_on_direct_rag_worker_plan():
    plan = build_fallback_plan("Summarize ARCHITECTURE.md.")

    assert len(plan) == 1
    assert plan[0]["executor"] == "rag_worker"
    assert plan[0]["mode"] == "sequential"
    assert plan[0]["result_mode"] == "answer"


def test_planner_routes_math_and_tabular_requests_to_specialists():
    utility_plan = build_fallback_plan("Calculate the monthly reserve from a 7% annual target.")
    analyst_plan = build_fallback_plan("Analyze this Excel spreadsheet and group revenue by region.")

    assert utility_plan[0]["executor"] == "utility"
    assert analyst_plan[0]["executor"] == "data_analyst"


def test_mixed_calculation_and_indexed_doc_search_decomposes_into_specialist_tasks():
    query = (
        "Calculate 18% of 4.2 million and, separately, search indexed docs for rate limit policy; "
        "return both results together."
    )

    plan = build_fallback_plan(query)

    assert [task["executor"] for task in plan] == ["utility", "rag_worker", "general"]
    assert [task["mode"] for task in plan[:2]] == ["parallel", "parallel"]
    assert plan[2]["depends_on"] == ["task_1", "task_2"]
    assert all("4.2" not in task["doc_scope"] for task in plan)


def test_normalise_task_plan_repairs_collapsed_mixed_calculation_and_retrieval_plan():
    query = (
        "Calculate 18% of 4.2 million and, separately, search indexed docs for rate limit policy; "
        "return both results together."
    )

    plan = normalise_task_plan(
        [
            {
                "id": "task_1",
                "title": "Handle request",
                "executor": "rag_worker",
                "mode": "sequential",
                "input": query,
                "doc_scope": ["4.2"],
            }
        ],
        query=query,
        max_tasks=8,
    )

    assert [task["executor"] for task in plan] == ["utility", "rag_worker", "general"]
    assert all("4.2" not in task["doc_scope"] for task in plan)


def test_planner_prompt_includes_context_packet_and_keeps_repair_safety_net():
    query = (
        "Calculate 18% of 4.2 million and, separately, search indexed docs for rate limit policy; "
        "return both results together."
    )
    chat = RecordingChatModel(
        """
        {
          "summary": "Collapsed plan",
          "tasks": [
            {
              "id": "task_1",
              "title": "Handle request",
              "executor": "rag_worker",
              "mode": "sequential",
              "input": "Calculate 18% of 4.2 million and search indexed docs for rate limit policy.",
              "doc_scope": ["4.2"],
              "skill_queries": []
            }
          ]
        }
        """
    )
    loop = QueryLoop(
        settings=SimpleNamespace(
            planner_max_tasks=4,
            memory_enabled=False,
            context_budget_enabled=False,
            tiktoken_enabled=False,
        ),
        providers=SimpleNamespace(chat=chat),
    )
    agent = AgentDefinition(
        name="planner",
        mode="planner",
        prompt_file="planner_agent.md",
    )
    packet = {
        "permission_mode": "default",
        "risk_flags": ["mixed_evidence_scopes"],
        "available_agents": ["planner", "utility", "rag_worker", "general"],
        "available_tools": ["calculator", "search_indexed_docs"],
    }

    result = loop.run(
        agent,
        SessionState(tenant_id="tenant", user_id="user", conversation_id="conv"),
        user_text=query,
        task_payload={"planner_input_packet": packet},
    )

    prompt = str(chat.calls[0]["messages"][-1].content)
    assert "PLANNER_CONTEXT:" in prompt
    assert '"permission_mode":"default"' in prompt
    assert '"available_agents":["planner","utility","rag_worker","general"]' in prompt
    assert result.metadata["planner_input_packet"] == packet
    assert result.metadata["plan_repair_applied"] is True
    assert [task["executor"] for task in result.metadata["planner_payload"]["tasks"]] == [
        "utility",
        "rag_worker",
        "general",
    ]


def test_graph_mutation_request_routes_to_general_admin_guidance():
    plan = build_fallback_plan("Import this graph and tell me whether it should be used for this request.")

    assert len(plan) == 1
    assert plan[0]["executor"] == "general"
    assert plan[0]["controller_hints"]["preferred_sources"] == ["graph"]


def test_graph_inventory_request_stays_on_general_catalog_path():
    plan = build_fallback_plan("What graphs do i have access to?")

    assert len(plan) == 1
    assert plan[0]["executor"] == "general"
    assert plan[0]["result_mode"] == "inventory"
    assert plan[0]["controller_hints"]["preferred_sources"] == ["graph"]
    assert plan[0]["controller_hints"]["prefer_inventory_output"] is True
    assert "list_graph_indexes" in plan[0]["input"]
    assert "search_indexed_docs" not in plan[0]["input"]


def test_graph_read_only_request_routes_to_graph_manager():
    plan = build_fallback_plan("What graphs exist, and should one of them be used for this request?")

    assert len(plan) == 1
    assert plan[0]["executor"] == "graph_manager"
    assert plan[0]["controller_hints"]["preferred_sources"] == ["graph"]
    assert plan[0]["controller_hints"]["prefer_graph"] is True


def test_dependency_ordering_enforces_sequential_work():
    task_plan = normalise_task_plan(
        [
            {
                "id": "task_1",
                "title": "Gather requirements",
                "executor": "rag_worker",
                "mode": "sequential",
                "depends_on": [],
                "input": "Find the requirements.",
            },
            {
                "id": "task_2",
                "title": "Summarize findings",
                "executor": "general",
                "mode": "sequential",
                "depends_on": ["task_1"],
                "input": "Summarize the extracted requirements.",
            },
        ],
        query="Find the requirements and summarize them.",
        max_tasks=8,
    )

    first_batch = select_execution_batch(task_plan, [])
    second_batch = select_execution_batch(
        task_plan,
        [{"task_id": "task_1", "status": "completed", "output": "done"}],
    )

    assert [task["id"] for task in first_batch] == ["task_1"]
    assert [task["id"] for task in second_batch] == ["task_2"]


def test_normalise_task_plan_repairs_collapsed_broad_grounded_synthesis_plan():
    query = (
        "Across the whole default collection, identify the major subsystems in this codebase, "
        "list the supporting documents for each subsystem, and verify that the final answer does not overclaim."
    )

    plan = normalise_task_plan(
        [
            {
                "id": "task_1",
                "title": "Handle request",
                "executor": "rag_worker",
                "mode": "sequential",
                "depends_on": [],
                "input": query,
                "doc_scope": [],
                "skill_queries": [],
            }
        ],
        query=query,
        max_tasks=8,
        session_metadata={"resolved_turn_intent": resolve_turn_intent(query, {}).to_dict()},
    )

    assert [task["id"] for task in plan] == ["task_1", "task_2", "task_3", "task_4", "task_5", "task_6", "task_7", "task_8"]
    assert plan[0]["produces_artifacts"] == ["title_candidates"]
    assert plan[1]["answer_mode"] == "evidence_only"
    assert plan[1]["produces_artifacts"] == ["doc_focus"]
    assert plan[2]["produces_artifacts"] == ["research_facets"]
    assert plan[4]["controller_hints"]["dynamic_triage_fanout"] is True
    assert plan[5]["controller_hints"]["dynamic_doc_review_fanout"] is True
    assert plan[6]["produces_artifacts"] == ["subsystem_inventory"]
    assert plan[7]["controller_hints"]["dynamic_subsystem_backfill"] is True


def test_normalise_task_plan_repairs_collapsed_holistic_repository_plan():
    query = "Investigate the major subsystems in this repo and give me a concise architectural walkthrough with citations."
    intent = resolve_turn_intent(query, {"kb_collection_id": "default"})

    plan = normalise_task_plan(
        [
            {
                "id": "task_1",
                "title": "Collect subsystem documentation",
                "executor": "rag_worker",
                "mode": "sequential",
                "depends_on": [],
                "input": "Identify major subsystems in the repository and retrieve their design docs, READMEs, architecture overviews.",
                "doc_scope": [],
                "skill_queries": [],
                "research_profile": "",
                "coverage_goal": "",
                "result_mode": "list",
                "answer_mode": "answer",
                "controller_hints": {},
                "produces_artifacts": ["subsystem_docs"],
                "handoff_schema": "subsystem_docs_schema",
            },
            {
                "id": "task_2",
                "title": "Synthesize architectural walkthrough",
                "executor": "general",
                "mode": "sequential",
                "depends_on": ["task_1"],
                "input": "subsystem_docs",
                "doc_scope": [],
                "skill_queries": [],
                "research_profile": "",
                "coverage_goal": "",
                "result_mode": "synthesis",
                "answer_mode": "answer",
                "controller_hints": {},
                "produces_artifacts": ["architectural_walkthrough"],
                "handoff_schema": "walkthrough_schema",
            },
        ],
        query=query,
        max_tasks=8,
        session_metadata={
            "kb_collection_id": "default",
            "resolved_turn_intent": intent.to_dict(),
        },
    )

    assert [task["id"] for task in plan] == ["task_1", "task_2", "task_3", "task_4", "task_5", "task_6", "task_7", "task_8"]
    assert plan[0]["produces_artifacts"] == ["title_candidates"]
    assert plan[2]["produces_artifacts"] == ["research_facets"]
    assert plan[4]["controller_hints"]["dynamic_triage_fanout"] is True
    assert plan[5]["controller_hints"]["dynamic_doc_review_fanout"] is True
    assert plan[6]["produces_artifacts"] == ["subsystem_inventory"]

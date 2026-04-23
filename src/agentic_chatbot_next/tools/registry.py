from __future__ import annotations

from typing import Any, Dict

from agentic_chatbot_next.contracts.tools import ToolDefinition
from agentic_chatbot_next.tools.groups.analyst import build_analyst_tools
from agentic_chatbot_next.tools.groups.discovery import build_discovery_tools
from agentic_chatbot_next.tools.groups.graph_gateway import build_graph_gateway_tools
from agentic_chatbot_next.tools.groups.memory import build_memory_tools
from agentic_chatbot_next.tools.groups.orchestration import build_orchestration_tools
from agentic_chatbot_next.tools.groups.rag_gateway import build_rag_gateway_tools
from agentic_chatbot_next.tools.groups.skills import build_skill_execution_tools
from agentic_chatbot_next.tools.groups.utility import build_utility_tools
from agentic_chatbot_next.tools.mcp_registry import build_mcp_tool_definitions


def _string_field(description: str, *, enum: list[str] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"type": "string", "description": description}
    if enum:
        payload["enum"] = list(enum)
    return payload


def _integer_field(description: str, *, minimum: int | None = None, maximum: int | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"type": "integer", "description": description}
    if minimum is not None:
        payload["minimum"] = minimum
    if maximum is not None:
        payload["maximum"] = maximum
    return payload


def _boolean_field(description: str) -> dict[str, Any]:
    return {"type": "boolean", "description": description}


def _array_field(description: str, *, item_type: str = "string") -> dict[str, Any]:
    return {
        "type": "array",
        "description": description,
        "items": {"type": item_type},
    }


def _object_schema(
    properties: dict[str, dict[str, Any]],
    *,
    required: list[str] | None = None,
    additional_properties: bool = False,
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(required or []),
        "additionalProperties": additional_properties,
    }


def _tool(
    *,
    name: str,
    group: str,
    builder: Any,
    description: str,
    args_schema: dict[str, Any],
    when_to_use: str,
    output_description: str,
    avoid_when: str = "",
    examples: list[str] | None = None,
    keywords: list[str] | None = None,
    read_only: bool = False,
    destructive: bool = False,
    background_safe: bool = False,
    concurrency_key: str = "",
    requires_workspace: bool = False,
    serializer: str = "default",
    should_defer: bool = False,
    search_hint: str = "",
    defer_reason: str = "",
    defer_priority: int = 50,
    eager_for_agents: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        group=group,
        builder=builder,
        description=description,
        args_schema=args_schema,
        when_to_use=when_to_use,
        avoid_when=avoid_when,
        output_description=output_description,
        examples=list(examples or []),
        keywords=list(keywords or []),
        read_only=read_only,
        destructive=destructive,
        background_safe=background_safe,
        concurrency_key=concurrency_key,
        requires_workspace=requires_workspace,
        serializer=serializer,
        should_defer=should_defer,
        search_hint=search_hint,
        defer_reason=defer_reason,
        defer_priority=defer_priority,
        eager_for_agents=list(eager_for_agents or []),
        metadata=dict(metadata or {}),
    )


_GRAPH_ADMIN_TOOLS = {
    "index_graph_corpus",
    "import_existing_graph",
    "refresh_graph_index",
}

_REQUIREMENTS_INVENTORY_TOOLS = {
    "extract_requirement_statements",
    "export_requirement_statements",
}


def _default_concurrency_key(definition: ToolDefinition) -> str:
    if definition.concurrency_key:
        return definition.concurrency_key
    if definition.name in _GRAPH_ADMIN_TOOLS:
        return "graph_admin"
    if definition.name in _REQUIREMENTS_INVENTORY_TOOLS:
        return "requirements_inventory"
    if definition.name == "rag_agent_tool":
        return "retrieval"
    if definition.group == "discovery":
        return "deferred_tool_discovery"
    if definition.group == "mcp":
        connection_id = str((definition.metadata or {}).get("connection_id") or "external")
        return f"mcp:{connection_id}"
    if definition.group == "memory":
        return "memory"
    if definition.group == "analyst":
        return "analyst_session"
    if definition.group == "orchestration":
        return "orchestration"
    if definition.group == "skills":
        return "skill_execution"
    return ""


def _apply_runtime_defaults(definitions: Dict[str, ToolDefinition]) -> Dict[str, ToolDefinition]:
    for definition in definitions.values():
        definition.concurrency_key = _default_concurrency_key(definition)
    return definitions


def build_tool_definitions(ctx: Any) -> Dict[str, ToolDefinition]:
    definitions = {
        "calculator": _tool(
            name="calculator",
            group="utility",
            builder=build_utility_tools,
            description="Evaluate arithmetic and unit-conversion expressions deterministically.",
            args_schema=_object_schema(
                {"expression": _string_field("Math expression to evaluate.")},
                required=["expression"],
            ),
            when_to_use="Use for arithmetic, percentages, rates, and unit conversions instead of mental math.",
            avoid_when="Avoid for document retrieval, symbolic programming, or tasks that need provenance beyond the expression itself.",
            output_description="Returns the computed numeric result or an evaluation error.",
            examples=["calculator(expression='2400 * 0.18')", "calculator(expression='5 * 1024')"],
            keywords=["math", "percentage", "conversion", "arithmetic"],
            read_only=True,
            background_safe=True,
            concurrency_key="utility",
        ),
        "list_indexed_docs": _tool(
            name="list_indexed_docs",
            group="utility",
            builder=build_utility_tools,
            description="List indexed documents, KB collections, or the documents accessible to the current session.",
            args_schema=_object_schema(
                {
                    "source_type": _string_field("Optional source filter.", enum=["", "kb", "upload"]),
                    "view": _string_field("Optional inventory view.", enum=["", "session_access", "kb_collections", "namespace_search"]),
                    "collection_id": _string_field("Optional collection id for KB inventory."),
                    "query": _string_field("Namespace query to search when view is namespace_search."),
                }
            ),
            when_to_use="Use first for inventory-style requests such as asking what documents, files, or KB collections are available.",
            avoid_when="Avoid when the user already named exact documents or wants grounded content instead of inventory.",
            output_description="Returns a JSON inventory payload describing accessible collections or documents.",
            examples=["list_indexed_docs(view='session_access')", "list_indexed_docs(source_type='kb')", "list_indexed_docs(view='namespace_search', query='rfp-corpus')"],
            keywords=["inventory", "documents", "knowledge base", "uploads", "namespace discovery"],
            read_only=True,
            background_safe=True,
        ),
        "search_skills": _tool(
            name="search_skills",
            group="utility",
            builder=build_utility_tools,
            description="Search the skills library for operating guidance, edge-case procedures, and reusable workflows.",
            args_schema=_object_schema(
                {
                    "query": _string_field("Natural-language query describing the guidance you need."),
                    "agent_filter": _string_field("Optional agent scope filter such as rag_agent or data_analyst_agent."),
                    "top_k": _integer_field("Maximum number of sections to return.", minimum=1, maximum=5),
                },
                required=["query"],
            ),
            when_to_use="Use when the runtime behavior is ambiguous and you want the curated procedure instead of improvising.",
            avoid_when="Avoid when the correct next step is already obvious from tool outputs or the user request.",
            output_description="Returns matching guidance sections from the skill library as text blocks.",
            examples=["search_skills(query='empty search results recovery')", "search_skills(query='dataset workflow', agent_filter='data_analyst_agent')"],
            keywords=["guidance", "workflow", "procedure", "skill lookup"],
            read_only=True,
            background_safe=True,
        ),
        "discover_tools": _tool(
            name="discover_tools",
            group="discovery",
            builder=build_discovery_tools,
            description="Search deferred tools that are allowed for the current agent but hidden from the initial tool list.",
            args_schema=_object_schema(
                {
                    "query": _string_field("Natural-language description of the capability or tool needed."),
                    "group": _string_field("Optional tool group filter."),
                    "top_k": _integer_field("Maximum number of deferred tools to return.", minimum=1, maximum=20),
                },
                required=["query"],
            ),
            when_to_use="Use when the current direct tools do not show a needed heavy or specialized capability, especially graph, plugin, MCP, or admin-style tools.",
            avoid_when="Avoid when a direct non-deferred tool already solves the task.",
            output_description="Returns matching deferred tool cards that can be invoked with call_deferred_tool.",
            examples=["discover_tools(query='search a graph for relationship evidence')"],
            keywords=["tool discovery", "deferred tools", "search tools", "hidden tools"],
            read_only=True,
            background_safe=True,
        ),
        "call_deferred_tool": _tool(
            name="call_deferred_tool",
            group="discovery",
            builder=build_discovery_tools,
            description="Invoke a deferred tool after it has been returned by discover_tools in the current turn.",
            args_schema=_object_schema(
                {
                    "tool_name": _string_field("Name of the deferred tool to invoke."),
                    "arguments": {
                        "type": "object",
                        "description": "Arguments to pass to the deferred target tool.",
                        "additionalProperties": True,
                    },
                },
                required=["tool_name", "arguments"],
                additional_properties=False,
            ),
            when_to_use="Use only after discover_tools returned the target tool and the deferred target is necessary for the task.",
            avoid_when="Avoid for direct tools, undiscovered tools, or attempts to bypass tool policy.",
            output_description="Returns a JSON envelope with target tool status, result, and safety metadata.",
            examples=["call_deferred_tool(tool_name='search_graph_index', arguments={'query': 'service dependencies'})"],
            keywords=["deferred tool", "invoke hidden tool", "tool proxy"],
            background_safe=True,
        ),
        "execute_skill": _tool(
            name="execute_skill",
            group="skills",
            builder=build_skill_execution_tools,
            description="Execute an active executable or hybrid skill from the runtime skills library.",
            args_schema=_object_schema(
                {
                    "skill_id": _string_field("Executable skill id to invoke."),
                    "input": _string_field("Natural-language task input for the skill."),
                    "arguments": {
                        "type": "object",
                        "description": "Optional structured arguments for the skill.",
                        "additionalProperties": True,
                    },
                },
                required=["skill_id"],
                additional_properties=False,
            ),
            when_to_use="Use when a searched or pinned executable skill directly matches the task and the current agent is allowed to run it.",
            avoid_when="Avoid for normal retrieval-only skill packs or when the current agent can answer directly without the reusable procedure.",
            output_description="Returns skill execution metadata and either inline instructions or a synchronous forked worker result.",
            examples=["execute_skill(skill_id='contract-review', input='Review uploaded MSA')"],
            keywords=["skill execution", "workflow", "procedure", "forked skill"],
            background_safe=True,
        ),
        "memory_save": _tool(
            name="memory_save",
            group="memory",
            builder=build_memory_tools,
            description="Persist a user-confirmed memory entry for later turns.",
            args_schema=_object_schema(
                {
                    "key": _string_field("Descriptive memory key."),
                    "value": _string_field("Fact or preference to store."),
                    "scope": _string_field("Memory scope.", enum=["conversation", "user"]),
                },
                required=["key", "value"],
            ),
            when_to_use="Use only for durable facts or preferences the user explicitly wants remembered.",
            avoid_when="Avoid for speculative notes, transient scratch work, or facts the user did not confirm.",
            output_description="Returns the saved memory record or an error if persistence is unavailable.",
            examples=["memory_save(key='preferred_region', value='us-east-1')"],
            keywords=["memory", "remember", "preference", "save"],
            background_safe=True,
        ),
        "memory_load": _tool(
            name="memory_load",
            group="memory",
            builder=build_memory_tools,
            description="Load a previously saved memory entry by key.",
            args_schema=_object_schema(
                {
                    "key": _string_field("Memory key to retrieve."),
                    "scope": _string_field("Memory scope.", enum=["conversation", "user"]),
                },
                required=["key"],
            ),
            when_to_use="Use when the user asks to recall a stored preference, fact, or prior decision.",
            avoid_when="Avoid guessing from conversation context when memory tools are available.",
            output_description="Returns the stored value for the requested key or a not-found result.",
            examples=["memory_load(key='preferred_region')"],
            keywords=["memory", "recall", "retrieve"],
            read_only=True,
            background_safe=True,
        ),
        "memory_list": _tool(
            name="memory_list",
            group="memory",
            builder=build_memory_tools,
            description="List saved memory keys available in the current scope.",
            args_schema=_object_schema(
                {"scope": _string_field("Memory scope.", enum=["conversation", "user"])}
            ),
            when_to_use="Use when the user asks what is remembered or when you need to inspect available memory keys before loading one.",
            avoid_when="Avoid when the user already named a specific key and you can call memory_load directly.",
            output_description="Returns the currently stored memory keys.",
            examples=["memory_list()"],
            keywords=["memory", "keys", "list"],
            read_only=True,
            background_safe=True,
        ),
        "rag_agent_tool": _tool(
            name="rag_agent_tool",
            group="rag_gateway",
            builder=build_rag_gateway_tools,
            description="Run grounded staged retrieval across the knowledge base and uploaded files, then return a stable answer contract.",
            args_schema=_object_schema(
                {
                    "query": _string_field("Grounded question to answer."),
                    "conversation_context": _string_field("Optional conversation or task context."),
                    "preferred_doc_ids_csv": _string_field("Optional comma-separated preferred doc ids."),
                    "must_include_uploads": _boolean_field("Whether uploaded files should be included when relevant."),
                    "top_k_vector": _integer_field("Vector retrieval candidate cap.", minimum=1),
                    "top_k_keyword": _integer_field("Keyword retrieval candidate cap.", minimum=1),
                    "max_retries": _integer_field("Retry budget for retrieval adjustment.", minimum=0),
                    "search_mode": _string_field("Search posture.", enum=["auto", "deep", "fast"]),
                    "max_search_rounds": _integer_field("Maximum retrieval rounds.", minimum=0),
                    "scratchpad_context_key": _string_field("Optional scratchpad key whose contents should augment conversation context."),
                },
                required=["query"],
            ),
            when_to_use="Use for grounded questions about indexed documents, uploaded files, contracts, policies, procedures, or any answer that needs citations.",
            avoid_when="Avoid for pure inventory questions or when exact named documents can be solved more directly with indexed-doc tools.",
            output_description="Returns the stable RAG contract as JSON, including answer text, citations, followups, and warnings.",
            examples=["rag_agent_tool(query='Summarize the incident response process.')"],
            keywords=["rag", "grounded answer", "citations", "documents"],
            background_safe=True,
        ),
        "resolve_indexed_docs": _tool(
            name="resolve_indexed_docs",
            group="rag_gateway",
            builder=build_rag_gateway_tools,
            description="Resolve one or more exact or near-exact indexed document names to stable doc ids.",
            args_schema=_object_schema(
                {"names": _array_field("List of document names or paths to resolve.")},
                required=["names"],
            ),
            when_to_use="Use when the user named specific indexed files and you need stable doc ids before reading or comparing them.",
            avoid_when="Avoid for broad discovery across the corpus; use search_indexed_docs first in that case.",
            output_description="Returns resolution results, including matched doc ids and any ambiguous or unresolved names.",
            examples=["resolve_indexed_docs(names=['api_rate_limits.md'])"],
            keywords=["resolve", "doc id", "file name", "exact document"],
            read_only=True,
            background_safe=True,
        ),
        "search_indexed_docs": _tool(
            name="search_indexed_docs",
            group="rag_gateway",
            builder=build_rag_gateway_tools,
            description="Search indexed document titles and source paths for likely candidate documents.",
            args_schema=_object_schema(
                {
                    "query": _string_field("Search query for document candidates."),
                    "collection_id": _string_field("Optional collection filter."),
                    "source_type": _string_field("Document source filter.", enum=["kb", "upload"]),
                    "limit": _integer_field("Maximum number of candidates to return.", minimum=1, maximum=50),
                },
                required=["query"],
            ),
            when_to_use="Use for title or path discovery before exact reads, comparisons, or scoped follow-up work.",
            avoid_when="Avoid when you already have the exact doc ids or when the user wants content-level synthesis rather than candidate discovery.",
            output_description="Returns candidate documents with titles, source paths, match reasons, and scores.",
            examples=["search_indexed_docs(query='incident response runbook')"],
            keywords=["search documents", "candidate docs", "title lookup"],
            read_only=True,
            background_safe=True,
        ),
        "read_indexed_doc": _tool(
            name="read_indexed_doc",
            group="rag_gateway",
            builder=build_rag_gateway_tools,
            description="Read an indexed document by overview, section, or paginated full-content mode.",
            args_schema=_object_schema(
                {
                    "doc_id": _string_field("Document id to read."),
                    "mode": _string_field("Read mode.", enum=["overview", "section", "full"]),
                    "focus": _string_field("Optional focus text for overview or full reads."),
                    "heading": _string_field("Optional section heading for section mode."),
                    "cursor": _integer_field("Chunk cursor for full reads.", minimum=0),
                    "max_chunks": _integer_field("Maximum chunks to return.", minimum=1, maximum=50),
                },
                required=["doc_id"],
            ),
            when_to_use="Use when you need direct document reads after narrowing to a specific indexed file.",
            avoid_when="Avoid for broad corpus questions where candidate discovery or staged RAG is a better first step.",
            output_description="Returns document metadata plus overview, section, or full-read chunks.",
            examples=["read_indexed_doc(doc_id='doc_123', mode='section', heading='Authentication')"],
            keywords=["read document", "section read", "full read"],
            read_only=True,
            background_safe=True,
        ),
        "compare_indexed_docs": _tool(
            name="compare_indexed_docs",
            group="rag_gateway",
            builder=build_rag_gateway_tools,
            description="Compare two indexed documents using deterministic reads from each side.",
            args_schema=_object_schema(
                {
                    "left_doc_id": _string_field("Left-hand document id."),
                    "right_doc_id": _string_field("Right-hand document id."),
                    "focus": _string_field("Optional comparison focus."),
                },
                required=["left_doc_id", "right_doc_id"],
            ),
            when_to_use="Use for direct file-vs-file comparisons after the documents are already resolved.",
            avoid_when="Avoid when the user asked for corpus-wide discovery or when the documents have not been resolved yet.",
            output_description="Returns shared and unique sections plus supporting evidence from each document.",
            examples=["compare_indexed_docs(left_doc_id='doc_a', right_doc_id='doc_b', focus='rate limits')"],
            keywords=["compare", "diff", "document comparison"],
            read_only=True,
            background_safe=True,
        ),
        "extract_requirement_statements": _tool(
            name="extract_requirement_statements",
            group="rag_gateway",
            builder=build_rag_gateway_tools,
            description="Extract statement-level mandatory requirements from uploads or one KB collection.",
            args_schema=_object_schema(
                {
                    "source_scope": _string_field("Document scope.", enum=["auto", "uploads", "kb"]),
                    "collection_id": _string_field("Optional KB collection id override."),
                    "document_names": _array_field("Optional exact document names or paths to resolve."),
                    "all_documents": _boolean_field("Whether to extract from every document in the selected scope."),
                    "mode": _string_field("Extraction mode.", enum=["mandatory", "strict_shall", "legal_clause"]),
                    "max_preview_rows": _integer_field("Maximum preview rows to return.", minimum=1, maximum=20),
                }
            ),
            when_to_use="Use for shall-statement inventories, mandatory-language extraction, or requirements harvesting from prose documents.",
            avoid_when="Avoid for spreadsheet/tabular analysis or for open-ended grounded Q&A that should use rag_agent_tool instead.",
            output_description="Returns counts, preview rows, source metadata, and ambiguity or unsupported-format feedback.",
            examples=["extract_requirement_statements(source_scope='uploads', all_documents=True, mode='strict_shall')"],
            keywords=["shall statements", "requirements extraction", "mandatory language", "verification matrix"],
            read_only=True,
            background_safe=True,
        ),
        "export_requirement_statements": _tool(
            name="export_requirement_statements",
            group="rag_gateway",
            builder=build_rag_gateway_tools,
            description="Export extracted requirement statements as a downloadable CSV artifact.",
            args_schema=_object_schema(
                {
                    "source_scope": _string_field("Document scope.", enum=["auto", "uploads", "kb"]),
                    "collection_id": _string_field("Optional KB collection id override."),
                    "document_names": _array_field("Optional exact document names or paths to resolve."),
                    "all_documents": _boolean_field("Whether to export every document in the selected scope."),
                    "mode": _string_field("Extraction mode.", enum=["mandatory", "strict_shall", "legal_clause"]),
                    "filename": _string_field("Optional CSV filename override."),
                    "max_preview_rows": _integer_field("Maximum preview rows to include in the tool result.", minimum=1, maximum=20),
                }
            ),
            when_to_use="Use when the user wants the full requirements inventory returned as a file or corpus-wide table export.",
            avoid_when="Avoid when a lightweight preview is enough and no artifact is needed.",
            output_description="Returns the extraction summary plus downloadable artifact metadata for the exported CSV.",
            examples=["export_requirement_statements(source_scope='kb', collection_id='rfp-corpus', all_documents=True, mode='strict_shall')"],
            keywords=["requirements export", "shall csv", "downloadable inventory"],
            requires_workspace=True,
            background_safe=True,
        ),
        "list_graph_indexes": _tool(
            name="list_graph_indexes",
            group="graph_gateway",
            builder=build_graph_gateway_tools,
            description="List managed graph indexes available to the current tenant.",
            args_schema=_object_schema(
                {
                    "collection_id": _string_field("Optional collection filter."),
                    "limit": _integer_field("Maximum number of graphs to return.", minimum=1, maximum=100),
                }
            ),
            when_to_use="Use before graph retrieval when you need to know what graph resources exist or are query-ready.",
            avoid_when="Avoid if the task already names the specific graph id and you can inspect it directly.",
            output_description="Returns graph metadata, readiness, and availability information.",
            examples=["list_graph_indexes(collection_id='default')"],
            keywords=["graph", "index", "catalog"],
            read_only=True,
            background_safe=True,
        ),
        "inspect_graph_index": _tool(
            name="inspect_graph_index",
            group="graph_gateway",
            builder=build_graph_gateway_tools,
            description="Inspect a managed graph index, including sources, readiness, and recent runs.",
            args_schema=_object_schema(
                {"graph_id": _string_field("Graph id to inspect.")},
                required=["graph_id"],
            ),
            when_to_use="Use when the request names an existing graph or when you need graph scope details before querying it.",
            avoid_when="Avoid for simple textual document lookup where graph state is not relevant.",
            output_description="Returns detailed graph metadata, source coverage, and recent run information.",
            examples=["inspect_graph_index(graph_id='security-graph')"],
            keywords=["graph", "inspect", "sources"],
            read_only=True,
            background_safe=True,
        ),
        "list_graph_documents": _tool(
            name="list_graph_documents",
            group="graph_gateway",
            builder=build_graph_gateway_tools,
            description="List the source documents recorded for a managed graph index.",
            args_schema=_object_schema(
                {"graph_id": _string_field("Graph id to inspect.")},
                required=["graph_id"],
            ),
            when_to_use="Use for graph inventory questions such as asking what documents or source files are in a named graph.",
            avoid_when="Avoid when the user needs graph relationships or query hits instead of source-document inventory.",
            output_description="Returns a JSON inventory payload describing the graph's recorded source documents.",
            examples=["list_graph_documents(graph_id='security-graph')"],
            keywords=["graph", "documents", "sources", "inventory"],
            read_only=True,
            background_safe=True,
            should_defer=True,
            search_hint="Use for deferred graph source-document inventory after a graph id is known.",
            defer_reason="Graph document inventories can be large and are only relevant to graph-specific work.",
            defer_priority=35,
            eager_for_agents=["graph_manager"],
        ),
        "search_graph_index": _tool(
            name="search_graph_index",
            group="graph_gateway",
            builder=build_graph_gateway_tools,
            description="Search one graph or a shortlist of relevant graphs for graph-backed evidence candidates.",
            args_schema=_object_schema(
                {
                    "query": _string_field("Graph search query."),
                    "graph_id": _string_field("Optional specific graph id."),
                    "collection_id": _string_field("Optional collection filter when graph_id is omitted."),
                    "methods_csv": _string_field("Optional comma-separated graph query methods."),
                    "top_k_graphs": _integer_field("Maximum graphs to search when graph_id is omitted.", minimum=1, maximum=8),
                    "limit": _integer_field("Maximum evidence items to return.", minimum=1, maximum=20),
                },
                required=["query"],
            ),
            when_to_use="Use for relationship-heavy, entity-centric, or multi-hop questions where graph structure may help.",
            avoid_when="Avoid when the user needs direct wording, exact quotations, or plain document inventory.",
            output_description="Returns graph-backed evidence candidates and the graph sources that supported them.",
            examples=["search_graph_index(query='Which services depend on the job manager?', methods_csv='local')"],
            keywords=["graph query", "relationships", "entities", "multi-hop"],
            read_only=True,
            background_safe=True,
            should_defer=True,
            search_hint="Use for graph-backed relationship, entity, dependency, network, or multi-hop evidence searches.",
            defer_reason="Graph search has a rich schema and should be loaded only when graph evidence is likely useful.",
            defer_priority=20,
            eager_for_agents=["graph_manager"],
        ),
        "explain_source_plan": _tool(
            name="explain_source_plan",
            group="graph_gateway",
            builder=build_graph_gateway_tools,
            description="Explain how the runtime would choose among graph, vector, keyword, and SQL-style sources.",
            args_schema=_object_schema(
                {
                    "query": _string_field("Question whose retrieval plan should be explained."),
                    "collection_id": _string_field("Optional collection filter."),
                    "preferred_doc_ids_csv": _string_field("Optional comma-separated preferred doc ids."),
                },
                required=["query"],
            ),
            when_to_use="Use when choosing the right retrieval lane is part of the task or when you want to justify graph-vs-text routing.",
            avoid_when="Avoid when you already know the source path and should just execute it.",
            output_description="Returns a structured explanation of likely source choices and why.",
            examples=["explain_source_plan(query='What changed in the release process?')"],
            keywords=["source planning", "routing", "graph vs vector"],
            read_only=True,
            background_safe=True,
            should_defer=True,
            search_hint="Use to compare graph, vector, keyword, and SQL retrieval lanes before choosing sources.",
            defer_reason="Source-planning guidance is specialized and not needed for most direct turns.",
            defer_priority=40,
            eager_for_agents=["graph_manager"],
        ),
        "index_graph_corpus": _tool(
            name="index_graph_corpus",
            group="graph_gateway",
            builder=build_graph_gateway_tools,
            description="Create or refresh a managed graph index for indexed source documents.",
            args_schema=_object_schema(
                {
                    "graph_id": _string_field("Optional stable graph id."),
                    "display_name": _string_field("Human-readable graph name."),
                    "collection_id": _string_field("Collection to index."),
                    "source_doc_ids_csv": _string_field("Optional comma-separated doc ids."),
                    "source_paths_csv": _string_field("Optional comma-separated source paths."),
                    "backend": _string_field("Optional graph backend override."),
                    "refresh": _boolean_field("Whether to refresh an existing graph."),
                }
            ),
            when_to_use="Use only in runtimes that explicitly allow graph lifecycle actions.",
            avoid_when="Avoid during normal end-user assistance or when graph administration is control-panel managed.",
            output_description="Returns graph indexing status and the resulting graph metadata.",
            examples=["index_graph_corpus(collection_id='default', display_name='Default corpus graph')"],
            keywords=["graph indexing", "graph build", "refresh graph"],
            destructive=True,
            background_safe=True,
            should_defer=True,
            search_hint="Use for explicit graph lifecycle requests to create or rebuild a managed graph index.",
            defer_reason="Graph lifecycle operations are heavy, admin-like, and potentially expensive.",
            defer_priority=90,
        ),
        "import_existing_graph": _tool(
            name="import_existing_graph",
            group="graph_gateway",
            builder=build_graph_gateway_tools,
            description="Register an existing graph artifact or Neo4j-backed graph in the managed catalog.",
            args_schema=_object_schema(
                {
                    "graph_id": _string_field("Optional stable graph id."),
                    "display_name": _string_field("Human-readable graph name."),
                    "collection_id": _string_field("Collection to associate."),
                    "import_backend": _string_field("Import backend.", enum=["neo4j", "artifact"]),
                    "artifact_path": _string_field("Path to the graph artifact when applicable."),
                    "source_doc_ids_csv": _string_field("Optional comma-separated source doc ids."),
                    "source_paths_csv": _string_field("Optional comma-separated source paths."),
                }
            ),
            when_to_use="Use only when importing a graph that already exists outside the managed runtime.",
            avoid_when="Avoid for normal search or when you simply need to inspect an existing managed graph.",
            output_description="Returns the registered graph metadata and import outcome.",
            examples=["import_existing_graph(import_backend='neo4j', graph_id='ops-graph')"],
            keywords=["graph import", "neo4j", "artifact registration"],
            destructive=True,
            background_safe=True,
            should_defer=True,
            search_hint="Use for explicit requests to register an existing Neo4j or artifact-backed graph.",
            defer_reason="Graph import is admin-like and should not be exposed during routine assistance.",
            defer_priority=95,
        ),
        "refresh_graph_index": _tool(
            name="refresh_graph_index",
            group="graph_gateway",
            builder=build_graph_gateway_tools,
            description="Refresh a managed graph index using its previously recorded sources.",
            args_schema=_object_schema(
                {"graph_id": _string_field("Graph id to refresh.")},
                required=["graph_id"],
            ),
            when_to_use="Use when a managed graph exists and its source data needs to be refreshed.",
            avoid_when="Avoid if graph refresh is admin-managed outside the runtime.",
            output_description="Returns the refresh status and updated graph metadata.",
            examples=["refresh_graph_index(graph_id='security-graph')"],
            keywords=["graph refresh", "reindex graph"],
            destructive=True,
            background_safe=True,
            should_defer=True,
            search_hint="Use for explicit requests to refresh or reindex an existing managed graph.",
            defer_reason="Graph refresh is an admin-like lifecycle operation.",
            defer_priority=90,
        ),
        "load_dataset": _tool(
            name="load_dataset",
            group="analyst",
            builder=build_analyst_tools,
            description="Load a CSV or Excel dataset from the knowledge base or session workspace into the analyst runtime.",
            args_schema=_object_schema(
                {
                    "doc_id": _string_field("Dataset reference or filename."),
                    "sheet_name": _string_field("Optional Excel sheet name."),
                }
            ),
            when_to_use="Use first in data-analysis workflows to understand the available dataset, columns, and shape.",
            avoid_when="Avoid if you already loaded the dataset and only need a quick scratchpad or workspace operation.",
            output_description="Returns schema, shape, dtypes, and a sample of the dataset.",
            examples=["load_dataset(doc_id='sales.csv')"],
            keywords=["dataset", "csv", "excel", "inspect data"],
            read_only=True,
            requires_workspace=True,
            background_safe=True,
        ),
        "inspect_columns": _tool(
            name="inspect_columns",
            group="analyst",
            builder=build_analyst_tools,
            description="Inspect selected dataset columns for nulls, types, distributions, and representative values.",
            args_schema=_object_schema(
                {
                    "doc_id": _string_field("Dataset reference or filename."),
                    "columns": _string_field("Comma-separated columns to inspect."),
                    "sheet_name": _string_field("Optional Excel sheet name."),
                }
            ),
            when_to_use="Use before coding to understand important columns, null patterns, and likely joins or aggregations.",
            avoid_when="Avoid if the user only asked for a file handoff or if the dataset inspection is already complete.",
            output_description="Returns per-column statistics and representative values.",
            examples=["inspect_columns(doc_id='sales.csv', columns='region,revenue')"],
            keywords=["columns", "statistics", "nulls", "dataset profiling"],
            read_only=True,
            requires_workspace=True,
            background_safe=True,
        ),
        "execute_code": _tool(
            name="execute_code",
            group="analyst",
            builder=build_analyst_tools,
            description="Execute Python analysis code in the sandboxed analyst environment.",
            args_schema=_object_schema(
                {
                    "code": _string_field("Python code to execute."),
                    "doc_ids": _string_field("Optional comma-separated dataset references that should be mounted."),
                },
                required=["code"],
            ),
            when_to_use="Use for pandas, statistics, plotting, or workbook logic after inspecting the dataset and planning the approach.",
            avoid_when="Avoid for simple arithmetic, bounded row-level NLP, or when you have not yet inspected the data.",
            output_description="Returns stdout, stderr, success status, and execution timing.",
            examples=["execute_code(code='print(2 + 2)')"],
            keywords=["python", "sandbox", "pandas", "analysis"],
            destructive=True,
            requires_workspace=True,
            background_safe=True,
        ),
        "run_nlp_column_task": _tool(
            name="run_nlp_column_task",
            group="analyst",
            builder=build_analyst_tools,
            description="Run a bounded LLM-powered NLP task over one text column in a dataset.",
            args_schema=_object_schema(
                {
                    "doc_id": _string_field("Dataset reference or filename."),
                    "sheet_name": _string_field("Optional Excel sheet name."),
                    "column": _string_field("Text column to process."),
                    "task": _string_field("NLP task.", enum=["sentiment", "categorize", "keywords", "summarize"]),
                    "classification_rules": _string_field("Optional classification instructions."),
                    "allowed_labels_csv": _string_field("Optional comma-separated allowed labels."),
                    "batch_size": _integer_field("Batch size for NLP execution.", minimum=1),
                    "output_mode": _string_field("Output mode.", enum=["summary_only", "append_columns"]),
                    "target_filename": _string_field("Optional output filename when appending columns."),
                    "label_column": _string_field("Optional label column name."),
                    "score_column": _string_field("Optional score column name."),
                }
            ),
            when_to_use="Use for row-level text labeling or bounded NLP instead of building ad hoc prompts inside execute_code.",
            avoid_when="Avoid for general dataframe logic, charting, or tasks that do not center on one text column.",
            output_description="Returns NLP summaries and, when requested, writes labeled output files into the workspace.",
            examples=["run_nlp_column_task(doc_id='tickets.csv', column='summary', task='categorize')"],
            keywords=["nlp", "classification", "sentiment", "keywords"],
            destructive=True,
            requires_workspace=True,
            background_safe=True,
        ),
        "return_file": _tool(
            name="return_file",
            group="analyst",
            builder=build_analyst_tools,
            description="Publish a workspace file as a downloadable artifact for the user.",
            args_schema=_object_schema(
                {
                    "filename": _string_field("Workspace filename to publish."),
                    "label": _string_field("Optional user-facing label."),
                }
            ),
            when_to_use="Use whenever analyst work produced a deliverable file that should be exposed in the response.",
            avoid_when="Avoid before the file exists in the session workspace.",
            output_description="Returns artifact registration metadata for the published file.",
            examples=["return_file(filename='sales__analyst_summary.xlsx', label='Analyzed workbook')"],
            keywords=["artifact", "download", "file handoff"],
            requires_workspace=True,
            background_safe=True,
        ),
        "scratchpad_write": _tool(
            name="scratchpad_write",
            group="analyst",
            builder=build_analyst_tools,
            description="Write a scratchpad note for the current session.",
            args_schema=_object_schema(
                {
                    "key": _string_field("Scratchpad key."),
                    "value": _string_field("Scratchpad value."),
                },
                required=["key", "value"],
            ),
            when_to_use="Use to store observations, plans, or intermediate findings inside the current session.",
            avoid_when="Avoid for durable user memory or when the information should instead be returned to the user immediately.",
            output_description="Returns the stored scratchpad entry.",
            examples=["scratchpad_write(key='analysis_plan', value='Inspect, aggregate, verify')"],
            keywords=["scratchpad", "notes", "intermediate state"],
            background_safe=True,
        ),
        "scratchpad_read": _tool(
            name="scratchpad_read",
            group="analyst",
            builder=build_analyst_tools,
            description="Read a scratchpad note from the current session.",
            args_schema=_object_schema(
                {"key": _string_field("Scratchpad key to read.")},
                required=["key"],
            ),
            when_to_use="Use to recover previously stored observations, plans, or interim results within the session.",
            avoid_when="Avoid when the value should come from durable memory or a workspace artifact instead.",
            output_description="Returns the stored scratchpad value or a not-found response.",
            examples=["scratchpad_read(key='analysis_plan')"],
            keywords=["scratchpad", "read notes"],
            read_only=True,
            background_safe=True,
        ),
        "scratchpad_list": _tool(
            name="scratchpad_list",
            group="analyst",
            builder=build_analyst_tools,
            description="List scratchpad keys currently stored in the session.",
            args_schema=_object_schema({}),
            when_to_use="Use when you need to inspect available scratchpad state before reading a specific key.",
            avoid_when="Avoid when you already know the exact scratchpad key to read.",
            output_description="Returns the available scratchpad keys.",
            examples=["scratchpad_list()"],
            keywords=["scratchpad", "list keys"],
            read_only=True,
            background_safe=True,
        ),
        "workspace_write": _tool(
            name="workspace_write",
            group="analyst",
            builder=build_analyst_tools,
            description="Write a text file into the persistent session workspace.",
            args_schema=_object_schema(
                {
                    "filename": _string_field("Workspace filename to write."),
                    "content": _string_field("Text content to write."),
                },
                required=["filename", "content"],
            ),
            when_to_use="Use to persist summaries, helper files, or follow-up artifacts across turns.",
            avoid_when="Avoid for binary files or when the content should remain ephemeral in scratchpad only.",
            output_description="Returns workspace write status and file metadata.",
            examples=["workspace_write(filename='analysis_summary.txt', content='Top findings...')"],
            keywords=["workspace", "write file", "persist"],
            requires_workspace=True,
            background_safe=True,
        ),
        "workspace_read": _tool(
            name="workspace_read",
            group="analyst",
            builder=build_analyst_tools,
            description="Read a text file from the persistent session workspace.",
            args_schema=_object_schema(
                {"filename": _string_field("Workspace filename to read.")},
                required=["filename"],
            ),
            when_to_use="Use to reopen summaries or text artifacts created earlier in the session workspace.",
            avoid_when="Avoid when the file should be handled as a dataset or downloadable artifact instead.",
            output_description="Returns the requested workspace file contents.",
            examples=["workspace_read(filename='analysis_summary.txt')"],
            keywords=["workspace", "read file"],
            read_only=True,
            requires_workspace=True,
            background_safe=True,
        ),
        "workspace_list": _tool(
            name="workspace_list",
            group="analyst",
            builder=build_analyst_tools,
            description="List files currently present in the session workspace.",
            args_schema=_object_schema({}),
            when_to_use="Use at the start of follow-up analyst work to see what files already exist in the persistent workspace.",
            avoid_when="Avoid when the next step already targets a known filename.",
            output_description="Returns the current workspace filenames.",
            examples=["workspace_list()"],
            keywords=["workspace", "list files"],
            read_only=True,
            requires_workspace=True,
            background_safe=True,
        ),
        "spawn_worker": _tool(
            name="spawn_worker",
            group="orchestration",
            builder=build_orchestration_tools,
            description="Spawn a scoped worker agent for delegated execution.",
            args_schema=_object_schema(
                {
                    "prompt": _string_field("Self-contained worker brief."),
                    "agent_name": _string_field("Worker agent to launch."),
                    "description": _string_field("Optional short description for job tracking."),
                    "run_in_background": _boolean_field("Whether the worker should continue in the background."),
                },
                required=["prompt"],
            ),
            when_to_use="Use when the task clearly needs scoped specialist execution or durable background work.",
            avoid_when="Avoid for simple direct work that the current agent can complete with its own tools.",
            output_description="Returns worker job metadata, including the new job id and launch status.",
            examples=["spawn_worker(prompt='Compare these two docs...', agent_name='coordinator')"],
            keywords=["worker", "delegate", "background job"],
        ),
        "message_worker": _tool(
            name="message_worker",
            group="orchestration",
            builder=build_orchestration_tools,
            description="Send a follow-up message to a running worker job.",
            args_schema=_object_schema(
                {
                    "job_id": _string_field("Worker job id."),
                    "message": _string_field("Follow-up instruction or clarification."),
                    "resume": _boolean_field("Whether the worker should resume immediately after receiving the message."),
                },
                required=["job_id", "message"],
            ),
            when_to_use="Use to continue a launched worker with new guidance or user follow-up context.",
            avoid_when="Avoid when there is no active worker job to continue.",
            output_description="Returns the message delivery status for the target worker job.",
            examples=["message_worker(job_id='job_123', message='Focus on API auth only.')"],
            keywords=["worker", "follow-up", "mailbox"],
        ),
        "request_parent_question": _tool(
            name="request_parent_question",
            group="orchestration",
            builder=build_orchestration_tools,
            description="Pause the current worker job and ask the parent or coordinator a typed question.",
            args_schema=_object_schema(
                {
                    "question": _string_field("Blocking question for the parent/coordinator."),
                    "reason": _string_field("Why the answer is needed."),
                    "options": _array_field("Optional answer choices."),
                    "context": _string_field("Relevant context for answering."),
                },
                required=["question"],
            ),
            when_to_use="Use inside a worker job when a missing answer materially blocks safe progress.",
            avoid_when="Avoid for soft ambiguity that can be handled with a reasonable assumption.",
            output_description="Returns the typed request id and marks the worker as waiting for a response.",
            examples=["request_parent_question(question='Which repository should I inspect?', options=['frontend','backend'])"],
            keywords=["worker question", "mailbox", "clarification"],
            background_safe=True,
        ),
        "request_parent_approval": _tool(
            name="request_parent_approval",
            group="orchestration",
            builder=build_orchestration_tools,
            description="Pause the current worker job and request operator approval for an action.",
            args_schema=_object_schema(
                {
                    "action": _string_field("Action requiring approval."),
                    "reason": _string_field("Why approval is needed."),
                    "tool_name": _string_field("Related tool, if any."),
                    "arguments": _object_schema({}),
                    "risk": _string_field("Risk or consequence of approving."),
                    "context": _string_field("Relevant context for the operator."),
                },
                required=["action", "reason"],
            ),
            when_to_use="Use inside a worker job before an action that needs explicit operator consent.",
            avoid_when="Avoid for actions already safely covered by existing instructions and tool policy.",
            output_description="Returns the typed approval request id and marks the worker as waiting.",
            examples=["request_parent_approval(action='Delete stale export file', reason='User asked to clean generated artifacts')"],
            keywords=["worker approval", "permission", "mailbox"],
            background_safe=True,
        ),
        "list_worker_requests": _tool(
            name="list_worker_requests",
            group="orchestration",
            builder=build_orchestration_tools,
            description="List typed worker question or approval requests for the current session.",
            args_schema=_object_schema(
                {
                    "job_id": _string_field("Optional worker job id."),
                    "status_filter": _string_field("Request status filter, usually open."),
                    "request_type": _string_field("Optional type filter such as question_request or approval_request."),
                },
            ),
            when_to_use="Use when a worker is waiting and you need to inspect its typed request.",
            avoid_when="Avoid when no worker jobs are involved.",
            output_description="Returns pending typed worker requests with job and request ids.",
            examples=["list_worker_requests(status_filter='open')"],
            keywords=["worker request", "mailbox", "approval", "question"],
            read_only=True,
            background_safe=True,
        ),
        "respond_worker_request": _tool(
            name="respond_worker_request",
            group="orchestration",
            builder=build_orchestration_tools,
            description="Answer an open worker question request and optionally resume the worker.",
            args_schema=_object_schema(
                {
                    "job_id": _string_field("Worker job id."),
                    "request_id": _string_field("Mailbox request id."),
                    "response": _string_field("Answer to deliver to the worker."),
                    "decision": _string_field("Reserved for operator approval API; agents must leave this empty."),
                    "resume": _boolean_field("Whether to resume the worker immediately."),
                },
                required=["job_id", "request_id", "response"],
            ),
            when_to_use="Use to answer question_request mailbox items for a worker you manage.",
            avoid_when="Avoid for approval_request items; those require operator/API approval.",
            output_description="Returns delivery status and the queued response message id.",
            examples=["respond_worker_request(job_id='job_123', request_id='msg_abc', response='Use the backend repo.')"],
            keywords=["worker response", "mailbox", "answer question"],
            background_safe=True,
        ),
        "invoke_agent": _tool(
            name="invoke_agent",
            group="orchestration",
            builder=build_orchestration_tools,
            description="Queue an async peer request for another allowed agent in the current session.",
            args_schema=_object_schema(
                {
                    "agent_name": _string_field("Allowed agent to invoke."),
                    "message": _string_field("Peer request message."),
                    "description": _string_field("Optional tracking description."),
                    "job_id": _string_field("Optional existing job id to reuse."),
                    "reuse_running_job": _boolean_field("Whether to reuse a compatible running job."),
                    "team_channel_id": _string_field("Optional team mailbox channel id to mirror this dispatch into."),
                },
                required=["agent_name", "message"],
            ),
            when_to_use="Use for bounded same-session peer follow-up when full coordinator orchestration is unnecessary.",
            avoid_when="Avoid when durable multi-worker planning or synthesis is needed; prefer spawn_worker with coordinator in that case.",
            output_description="Returns the queued peer request status and job metadata.",
            examples=["invoke_agent(agent_name='utility', message='List the current KB files.')"],
            keywords=["peer agent", "async request", "follow-up"],
            background_safe=True,
        ),
        "create_team_channel": _tool(
            name="create_team_channel",
            group="orchestration",
            builder=build_orchestration_tools,
            description="Create a same-session team mailbox channel for async multi-agent coordination.",
            args_schema=_object_schema(
                {
                    "name": _string_field("Short channel name."),
                    "purpose": _string_field("Why the channel exists."),
                    "member_agents": _array_field("Allowed peer agents to include."),
                    "member_job_ids": _array_field("Existing job ids to include."),
                },
                required=["name"],
            ),
            when_to_use="Use when several allowed agents need a shared async coordination thread.",
            avoid_when="Avoid for one-off parent/worker clarification; use worker request tools instead.",
            output_description="Returns the created team mailbox channel metadata.",
            examples=["create_team_channel(name='contract-review', member_agents=['rag_worker','utility'])"],
            keywords=["team mailbox", "channel", "swarm", "coordination"],
            background_safe=True,
        ),
        "post_team_message": _tool(
            name="post_team_message",
            group="orchestration",
            builder=build_orchestration_tools,
            description="Post a typed message into a team mailbox channel.",
            args_schema=_object_schema(
                {
                    "channel_id": _string_field("Team mailbox channel id."),
                    "content": _string_field("Message body."),
                    "message_type": _string_field("Typed message kind."),
                    "target_agents": _array_field("Optional target agent names."),
                    "target_job_ids": _array_field("Optional target job ids."),
                    "subject": _string_field("Optional short subject."),
                    "payload": _object_schema({}),
                },
                required=["channel_id", "content"],
            ),
            when_to_use="Use for async handoffs, status updates, or non-blocking team questions in an existing channel.",
            avoid_when="Avoid for blocking worker questions that should pause the worker.",
            output_description="Returns the posted team mailbox message metadata.",
            examples=["post_team_message(channel_id='tmc_123', message_type='handoff', content='Evidence summary ready')"],
            keywords=["team mailbox", "handoff", "status", "question"],
            background_safe=True,
        ),
        "list_team_messages": _tool(
            name="list_team_messages",
            group="orchestration",
            builder=build_orchestration_tools,
            description="List team mailbox channel messages visible to the current agent.",
            args_schema=_object_schema(
                {
                    "channel_id": _string_field("Optional team mailbox channel id."),
                    "message_type": _string_field("Optional message type filter."),
                    "status_filter": _string_field("Optional status filter, usually open."),
                    "limit": _integer_field("Maximum messages to return.", minimum=1, maximum=200),
                }
            ),
            when_to_use="Use to inspect open team handoffs, questions, approvals, and status updates.",
            avoid_when="Avoid when no team channel is involved.",
            output_description="Returns team mailbox messages and channel summaries.",
            examples=["list_team_messages(channel_id='tmc_123', status_filter='open')"],
            keywords=["team mailbox", "list messages", "handoff", "approval"],
            read_only=True,
            background_safe=True,
        ),
        "claim_team_messages": _tool(
            name="claim_team_messages",
            group="orchestration",
            builder=build_orchestration_tools,
            description="Claim open team mailbox messages for the current agent/job.",
            args_schema=_object_schema(
                {
                    "channel_id": _string_field("Team mailbox channel id."),
                    "limit": _integer_field("Maximum messages to claim.", minimum=1, maximum=50),
                    "message_type": _string_field("Optional message type filter."),
                },
                required=["channel_id"],
            ),
            when_to_use="Use when the agent is ready to consume async team handoffs or questions.",
            avoid_when="Avoid if the agent only needs to inspect without taking ownership.",
            output_description="Returns claimed messages; request messages remain open until answered.",
            examples=["claim_team_messages(channel_id='tmc_123', message_type='handoff')"],
            keywords=["team mailbox", "claim", "consume", "handoff"],
            background_safe=True,
        ),
        "respond_team_message": _tool(
            name="respond_team_message",
            group="orchestration",
            builder=build_orchestration_tools,
            description="Answer an open team mailbox question request.",
            args_schema=_object_schema(
                {
                    "channel_id": _string_field("Team mailbox channel id."),
                    "message_id": _string_field("Question request message id."),
                    "response": _string_field("Answer to post."),
                    "decision": _string_field("Reserved for operator approval API; agents must leave this empty."),
                    "resolve": _boolean_field("Whether to resolve the request."),
                },
                required=["channel_id", "message_id", "response"],
            ),
            when_to_use="Use to answer question_request messages in a team channel.",
            avoid_when="Avoid for approval_request items; those require operator/API approval.",
            output_description="Returns request and response message metadata.",
            examples=["respond_team_message(channel_id='tmc_123', message_id='tmm_abc', response='Use default collection')"],
            keywords=["team mailbox", "response", "answer question"],
            background_safe=True,
        ),
        "list_jobs": _tool(
            name="list_jobs",
            group="orchestration",
            builder=build_orchestration_tools,
            description="List worker jobs for the current session.",
            args_schema=_object_schema(
                {"status_filter": _string_field("Optional job status filter.")}
            ),
            when_to_use="Use when the user asks about active or historical delegated work or when you need to inspect job state.",
            avoid_when="Avoid when there are no delegated jobs involved in the current task.",
            output_description="Returns session job metadata, statuses, and summaries.",
            examples=["list_jobs(status_filter='running')"],
            keywords=["jobs", "status", "background work"],
            read_only=True,
        ),
        "stop_job": _tool(
            name="stop_job",
            group="orchestration",
            builder=build_orchestration_tools,
            description="Stop a running or queued background job.",
            args_schema=_object_schema(
                {"job_id": _string_field("Job id to stop.")},
                required=["job_id"],
            ),
            when_to_use="Use when background work is no longer needed or the user explicitly wants it cancelled.",
            avoid_when="Avoid unless stopping the job is intentional; this is a mutating orchestration action.",
            output_description="Returns the cancellation outcome for the target job.",
            examples=["stop_job(job_id='job_123')"],
            keywords=["cancel job", "stop worker"],
        ),
    }
    definitions.update(build_mcp_tool_definitions(ctx))
    return _apply_runtime_defaults(definitions)

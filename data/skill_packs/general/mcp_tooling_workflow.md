---
name: MCP Tooling Workflow
agent_scope: general
tool_tags: search_skills, discover_tools, call_deferred_tool, mcp__*
task_tags: mcp, external_tools, tool_discovery
version: 1
enabled: true
description: Use registered MCP servers through deferred discovery without falling back to KB documents as if they were live external results.
keywords: mcp, model context protocol, external server, tool discovery, plugin, remote tool
when_to_apply: Use when the user asks to use MCP tooling or a registered MCP server/service, or when the route context says MCP intent was detected.
avoid_when: Avoid for pure conceptual questions that do not ask to use, list, or inspect available MCP services.
examples: use MCP tooling, search SAM.gov with MCP, list available MCP tools
---
# MCP Tooling Workflow

## Workflow

Treat MCP tools as the primary evidence path for MCP-intent turns.

1. Preserve the user's task as the discovery query.
2. Call `discover_tools(query=<user task>, group="mcp")` before any MCP invocation.
3. Choose the returned MCP tool that best matches the user's requested service and action. If multiple tools are plausible and the choice changes the result, ask a concise clarification.
4. Call `call_deferred_tool` with one of the MCP tool names returned by discovery in this same turn.
5. Summarize the MCP result in the user's requested format and include useful external links or identifiers returned by the tool.

## Failure Recovery

- If discovery returns no MCP matches, report that no matching MCP tool is currently available and suggest checking the control-panel connection, allowed agents, visibility, and tool catalog.
- If the MCP call returns no results, report the filters or arguments used and suggest safe ways to broaden them.
- If the MCP server returns an auth, rate-limit, validation, or upstream error, report the failure class without exposing secrets.
- Do not answer from repo docs, setup notes, or RAG search as a substitute for live MCP results when the user explicitly asked to use MCP.

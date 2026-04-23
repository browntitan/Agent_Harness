# Acme Agent Platform — Integrations & Tool Calling

**Last updated:** 2026-01-18

## 1. Tool calling overview

Tools are the primary way agents interact with systems.

Common tools:

- calculator / unit conversion
- HTTP API calls
- database queries
- ticketing integrations
- internal RAG agent

## 2. Tool definition best practices

### 2.1 Keep schemas simple

Prefer arguments like:

- strings
- numbers
- booleans
- arrays of strings

Avoid:

- deeply nested objects
- unions
- optional fields with complex constraints

### 2.2 Describe when to use the tool

A tool description should include:

- what the tool does
- when it should be used
- what not to do

Example (good):

> "Use this tool to search internal runbooks and return evidence with citations."

## 3. Tool calling error handling

AAP recommends agents follow a consistent pattern:

1) Make a short plan
2) Call tools one at a time
3) After each tool result, check:
   - did it succeed?
   - is output complete?
4) If failures occur:
   - retry with fixed arguments
   - or ask user for clarification

## 4. Tool calling budgets

Set budgets to prevent runaway loops:

- max tool calls per turn
- max total steps per turn

If budgets are exceeded:

- return partial results
- ask user how to proceed

## 5. Building a RAG agent tool

AAP recommends implementing RAG as a specialist tool with this behavior:

- fan-out retrieval (vector + keyword)
- relevance grading
- query rewriting
- grounded answer generation
- citations

This is sometimes called **agentic RAG**.

## 6. Integration examples

### 6.1 Ticketing

- create_ticket(title, description, severity)
- update_ticket(id, comment)

### 6.2 Monitoring

- query_metrics(metric_name, time_range)
- get_incident_timeline(incident_id)

### 6.3 Knowledge search

- rag_agent_tool(question, context)


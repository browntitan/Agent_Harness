# Acme Agent Platform — Product Overview (v1.4)

**Last updated:** 2026-01-12

## 1. What is the Acme Agent Platform?

The **Acme Agent Platform (AAP)** is an orchestration runtime for building agentic applications.
It supports:

- Tool calling (function-style tools)
- Multi-agent workflows (specialists + orchestrators)
- Retrieval-augmented generation (RAG) pipelines
- Observability hooks for tracing tool calls and intermediate steps

AAP is designed for both:

- **interactive chatbots** (end-user UX)
- **back-office automations** (batch or event-driven)

## 2. Core components

### 2.1 Router
A router selects between:

- a low-latency "direct response" path
- a higher-latency "agent" path with tools and retrieval

Recommended routing inputs:

- whether the user uploaded files
- whether the user asked for citations
- whether a tool is required

### 2.2 Orchestrator agent
The orchestrator is a general-purpose agent that:

- interprets the user request
- makes a plan for tool usage (if needed)
- calls tools sequentially
- synthesizes outputs into a final response

### 2.3 Specialist agents
Specialists are narrow agents used as tools.

Common specialists:

- **RAG agent**: search + ground answers with citations
- **Code agent**: run tests, reason about diffs
- **Policy agent**: interpret policy and compliance requirements

## 3. Tool calling model

AAP defines tools as:

- name
- description
- JSON schema for arguments
- callable implementation

### 3.1 Tool calling reliability guidelines

1) Prefer tools with **narrow scope**
2) Prefer **simple argument schemas**
3) Validate tool outputs before using them
4) Add budgets:
   - max tool calls per turn
   - max steps per turn

## 4. RAG in AAP

AAP supports multiple retrieval strategies:

- vector similarity
- keyword search (BM25)
- metadata filters (doc type, date, tags)

AAP recommends "agentic RAG" when:

- questions are ambiguous
- the corpus is large or noisy
- citations are required

Agentic RAG typically includes:

- relevance grading of retrieved chunks
- query rewriting if evidence is weak
- grounded generation

## 5. Intended use cases

- **Enterprise internal assistant** grounded in policies, runbooks, and specs
- **Support bot** that can consult troubleshooting guides
- **Developer assistant** that can call tools (calculators, APIs, repo search)

## 6. Known limitations

- Tool calling quality depends on the LLM.
- Retrieval quality depends on chunking and indexing.

AAP recommends:

- using a judge model for relevance grading
- keeping prompts/tool descriptions short and explicit


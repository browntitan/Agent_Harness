---
name: Graph Grounding And Resolve Back
agent_scope: rag
tool_tags: search_graph_index, read_indexed_doc, rag_agent_tool
task_tags: graph, grounding, resolve_back
version: 2
enabled: true
description: Ground graph-discovered entities or relationships back into retrieved document text before final synthesis.
keywords: graph grounding, resolve back, evidence
when_to_apply: Use whenever graph search identifies useful leads that need textual confirmation.
avoid_when: Avoid answering solely from graph structure when exact wording matters.
examples: relationship confirmation, entity attribute grounding
---
# Graph Grounding And Resolve Back

## Rule

Graph search can suggest the answer path, but direct text evidence should support the final grounded claim whenever the claim is document-specific.

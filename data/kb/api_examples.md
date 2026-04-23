# Internal API — Examples

## 1. List agents

```bash
curl -H "Authorization: Bearer $API_KEY" \
  https://api.acme.internal/v1/agents
```

## 2. Create agent

```bash
curl -X POST \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "SupportBot",
    "description": "Grounded support assistant",
    "tools": ["rag_agent_tool", "calculator"]
  }' \
  https://api.acme.internal/v1/agents
```

## 3. Upload document

```bash
curl -X POST \
  -H "Authorization: Bearer $API_KEY" \
  -F "file=@./incident_postmortem.pdf" \
  -F 'metadata={"type": "postmortem", "team": "sre"}' \
  https://api.acme.internal/v1/documents
```


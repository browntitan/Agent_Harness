# Internal API — Endpoints

**Base URL:** `https://api.acme.internal`

## 1. Agents

### 1.1 List agents

`GET /v1/agents`

Query params:

- `limit` (int, default 50)
- `cursor` (string)

### 1.2 Get agent

`GET /v1/agents/{agent_id}`

### 1.3 Create agent

`POST /v1/agents`

Body:

- `name` (string)
- `description` (string)
- `tools` (array)

### 1.4 Update agent

`PATCH /v1/agents/{agent_id}`

## 2. Runs

### 2.1 Create run

`POST /v1/runs`

Body:

- `agent_id` (string)
- `input` (object)

### 2.2 Get run

`GET /v1/runs/{run_id}`

### 2.3 List run steps

`GET /v1/runs/{run_id}/steps`

## 3. Documents

### 3.1 Upload document

`POST /v1/documents`

Multipart form:

- `file`
- `metadata` (JSON)

### 3.2 List documents

`GET /v1/documents`

## 4. Observability

### 4.1 List traces

`GET /v1/traces`


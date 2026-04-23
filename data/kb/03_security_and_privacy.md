# Acme Agent Platform — Security & Privacy

**Last updated:** 2026-01-20

## 1. Data classification

AAP supports three data classes:

- Public
- Internal
- Restricted

Restricted data examples:

- customer PII
- credentials and secrets
- regulated health data

## 2. Encryption

- Data in transit: TLS 1.2+
- Data at rest: AES-256

## 3. Data retention

Retention depends on the plan and configuration.

### 3.1 Free
- 7 days conversation retention
- 7 days trace retention

### 3.2 Pro
- 30 days conversation retention
- 30 days trace retention

### 3.3 Enterprise
Enterprise retention is configurable.
Defaults are 365 days.

## 4. Document uploads

Uploaded documents are treated as **knowledge sources**.

### 4.1 Upload lifecycle

1) Upload is stored and indexed
2) Chunked text is embedded
3) Embeddings and chunk text are stored

### 4.2 Upload deletion

If a document is deleted:

- chunk text is removed
- embeddings are removed
- citations referencing that document become invalid

## 5. Prompt injection considerations

When building tool agents, AAP recommends:

- treat retrieved text as untrusted
- never execute tool calls directly from retrieved content
- allow-list tools
- implement budgets

## 6. Logging

AAP logs:

- tool call name
- tool call arguments (configurable redaction)
- tool outputs (optional)
- timing and errors

For restricted data, enable argument/output redaction.

## 7. Compliance

AAP Enterprise supports:

- audit logs
- role-based access control
- SSO
- data residency (add-on)


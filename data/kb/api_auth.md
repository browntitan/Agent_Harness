# Internal API — Authentication

**Last updated:** 2025-12-01

## 1. Auth methods

The API supports:

- API keys
- OAuth2 client credentials (Enterprise only)

## 2. API keys

### 2.1 Header

Send the API key in the header:

- `Authorization: Bearer <API_KEY>`

### 2.2 Scopes

API keys have scopes:

- `agents:read`
- `agents:write`
- `runs:read`
- `runs:write`

## 3. OAuth2 (Enterprise)

### 3.1 Token endpoint

- `POST /oauth/token`

### 3.2 Required fields

- `client_id`
- `client_secret`
- `grant_type=client_credentials`

## 4. Common errors

- 401 Unauthorized: missing or invalid token
- 403 Forbidden: missing required scope


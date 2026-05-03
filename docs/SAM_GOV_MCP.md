# SAM.gov MCP Control-Panel Test

This walkthrough validates the existing MCP control-panel path with SAM.gov contract opportunities.

## Prerequisites

Set a live SAM.gov API key in `.env`:

```env
SAM_API_KEY=your-sam-api-key
MCP_TOOL_PLANE_ENABLED=true
DEFERRED_TOOL_DISCOVERY_ENABLED=true
MCP_REQUIRE_HTTPS=false
MCP_ALLOW_PRIVATE_NETWORK=true
MCP_SECRET_ENCRYPTION_KEY=change-me-local-mcp-secret
```

Use HTTPS-only MCP URLs outside local development.

## Option A: Owned SAM.gov MCP Server

Start the normal stack plus the owned Streamable HTTP MCP server:

```bash
docker compose --profile sam-gov-mcp up -d --build
```

Register the connection in the control panel at `http://127.0.0.1:18000/control-panel`:

- Display Name: `sam-gov`
- Server URL: `http://sam-gov-mcp:8000/mcp`
- Bearer Token: blank
- Allowed Agents: `general`
- Visibility: `Tenant`
- Managed Service: `sam-gov-mcp`
- Server Environment: `SAM_API_KEY=your-sam-api-key`

Then click `Add MCP`, `Restart Server`, `Test`, `Refresh Tools`, and verify these tools appear:

- `mcp__sam_gov__search_open_contracts`
- `mcp__sam_gov__get_opportunity_description`

The owned server marks both tools read-only through MCP tool annotations, so the catalog should show `Read-only` after refresh.

## Option B: Open-Source Smoke Test

The `rsivilli/sam-mcp` package currently requires Python 3.13, so it runs in a separate smoke-test profile rather than the app image.

```bash
docker compose --profile sam-mcp-smoke up -d sam-mcp-smoke
```

Register it with:

- Display Name: `sam-gov-smoke`
- Server URL: `http://sam-mcp-smoke:8000/mcp`
- Bearer Token: blank
- Allowed Agents: `general`
- Visibility: `Tenant`
- Managed Service: `sam-mcp-smoke`
- Server Environment: `SAM_API_KEY=your-sam-api-key`

Use this only to prove the control-panel MCP lifecycle against a third-party server. Keep the owned server for repeatable local demos and controlled output.

## Environment And Restart Behavior

The control panel can now store MCP server environment variables on the connection. Values are encrypted in backend metadata and the UI only shows configured key names after saving. Leaving the Server Environment editor blank preserves the current encrypted values; entering `KEY=value` lines replaces them.

You do not need to restart the main application after editing MCP connection metadata. You do need to restart or recreate the MCP server process before new environment variables affect the server. The `Restart Server` button works only for managed Docker Compose sidecars whose connection metadata declares a `server_runtime` service, such as `sam-gov-mcp` or `sam-mcp-smoke`.

Bring-your-own MCP servers do not have to be installed in the same Python environment as the app. They only need to expose a reachable Streamable HTTP `/mcp` endpoint. If they run elsewhere, restart them in their own host, container platform, or cloud runtime.

## Runtime Flow

1. The control panel creates a row in `mcp_connections`.
2. `Test` calls remote MCP `list_tools`.
3. `Refresh Tools` caches discovered tools in `mcp_tool_catalog`.
4. The runtime exposes cached rows as dynamic `mcp__...` tool definitions.
5. Because deferred discovery is enabled, the model first calls `discover_tools`.
6. The model calls `call_deferred_tool` only after the SAM.gov tool is returned by discovery.
7. The MCP client invokes the remote Streamable HTTP server.

Expected events include `deferred_tool_discovery_searched`, `deferred_tool_invoked`, and `mcp_tool_invoked`.

## Example Prompts

```text
Find 5 active SDVOSB set-aside solicitations for IT services in SAM.gov posted from 04/02/2026 to 05/02/2026. Prefer NAICS 541512. Include notice ID, title, agency, response deadline, set-aside, and SAM link.
```

Expected owned-server call:

```json
{
  "posted_from": "04/02/2026",
  "posted_to": "05/02/2026",
  "keyword": "IT services",
  "naics_code": "541512",
  "set_aside_code": "SDVOSBC",
  "limit": 5,
  "active_only": true
}
```

```text
Use the SAM.gov MCP tool to get the full description for notice ID <notice_id> and summarize the work, deadline, and submission instructions.
```

Expected owned-server call:

```json
{
  "notice_id": "<notice_id>",
  "include_html": false
}
```

If SAM.gov returns no description, the tool returns `status: "not_found"` and the agent should say that directly.

## Troubleshooting

- `MCP tool plane is disabled`: set `MCP_TOOL_PLANE_ENABLED=true` and restart `app`.
- HTTP URL rejected: set `MCP_REQUIRE_HTTPS=false` and `MCP_ALLOW_PRIVATE_NETWORK=true` for local development.
- Empty catalog: verify the server URL is reachable from the `app` container, not just the host.
- 401 or 403 from SAM.gov: regenerate the SAM.gov API key and restart the MCP sidecar.
- 429 from SAM.gov: wait for rate limits to reset or use a higher-limit SAM.gov key.

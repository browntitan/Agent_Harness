# Web Research MCP Control-Panel Test

This walkthrough validates the owned Web Research MCP sidecar for public web search and public website parsing.

## Prerequisites

Create a Tavily API key at `https://app.tavily.com` and set it in `.env`:

```env
TAVILY_API_KEY=your-tavily-api-key
JINA_API_KEY=
MCP_TOOL_PLANE_ENABLED=true
DEFERRED_TOOL_DISCOVERY_ENABLED=true
MCP_REQUIRE_HTTPS=false
MCP_ALLOW_PRIVATE_NETWORK=true
MCP_SECRET_ENCRYPTION_KEY=change-me-local-mcp-secret
WEB_RESEARCH_TIMEOUT_SECONDS=30
WEB_RESEARCH_MAX_CONTENT_CHARS=20000
WEB_RESEARCH_ALLOW_PRIVATE_NETWORK=false
```

`JINA_API_KEY` is optional. Jina Reader can read many public pages without a key, but a key can improve quota and reliability.

Keep `WEB_RESEARCH_ALLOW_PRIVATE_NETWORK=false` unless you are deliberately testing against private-network pages. The server rejects localhost, private IP, link-local, reserved, credentialed, and non-HTTP(S) URLs by default.

## Start The Sidecar

Start the normal stack plus the owned Streamable HTTP MCP server:

```bash
docker compose --profile web-research-mcp up -d --build
```

The host endpoint is:

```text
http://127.0.0.1:18082/mcp
```

From inside the Docker Compose network, register:

```text
http://web-research-mcp:8000/mcp
```

## Register In The Control Panel

Open `http://127.0.0.1:18000/control-panel` and add an MCP connection:

- Display Name: `web-research`
- Server URL: `http://web-research-mcp:8000/mcp`
- Bearer Token: blank
- Allowed Agents: `general`
- Visibility: `Tenant`
- Managed Service: `web-research-mcp`
- Server Environment:

```env
TAVILY_API_KEY=your-tavily-api-key
JINA_API_KEY=optional-jina-key
```

Then click `Add MCP`, `Restart Server`, `Test`, and `Refresh Tools`.

Verify these tools appear:

- `mcp__web_research__web_search`
- `mcp__web_research__read_url`
- `mcp__web_research__batch_read_urls`
- `mcp__web_research__search_and_read`

All tools are read-only. The tool catalog should show `Read-only` after refresh.

## Runtime Flow

1. The control panel stores the connection in `mcp_connections`.
2. `Test` calls remote MCP `list_tools`.
3. `Refresh Tools` caches discovered tools in `mcp_tool_catalog`.
4. The runtime exposes cached rows as dynamic `mcp__...` tool definitions.
5. Deferred tool discovery lets the model find the web research tools only when the user asks for current public-web information or page parsing.
6. The MCP client invokes the remote Streamable HTTP server only on an actual tool call.

Expected events include `deferred_tool_discovery_searched`, `deferred_tool_invoked`, and `mcp_tool_invoked`.

## Example Prompts

```text
Use the web research MCP tool to search for recent guidance on MCP Streamable HTTP servers. Return 5 sources with titles, URLs, snippets, and publication dates when available.
```

Expected call:

```json
{
  "query": "recent guidance on MCP Streamable HTTP servers",
  "max_results": 5,
  "topic": "general",
  "include_answer": false
}
```

```text
Use the web research MCP tool to read https://modelcontextprotocol.io/specification/2025-03-26/basic/transports and summarize the Streamable HTTP transport.
```

Expected call:

```json
{
  "url": "https://modelcontextprotocol.io/specification/2025-03-26/basic/transports",
  "provider": "auto",
  "format": "markdown",
  "max_chars": 20000
}
```

```text
Search the web for Tavily Extract API docs and read the top 3 results. Summarize the supported request parameters.
```

Expected call:

```json
{
  "query": "Tavily Extract API docs supported request parameters",
  "max_results": 5,
  "read_top_n": 3
}
```

## Troubleshooting

- `MCP tool plane is disabled`: set `MCP_TOOL_PLANE_ENABLED=true` and restart `app`.
- HTTP URL rejected by the control panel: set `MCP_REQUIRE_HTTPS=false` and `MCP_ALLOW_PRIVATE_NETWORK=true` for local development.
- `TAVILY_API_KEY is not configured`: add the key to the Web Research MCP server environment and restart `web-research-mcp`.
- `Tavily rejected the configured API key`: regenerate the key in Tavily and restart the MCP sidecar.
- `Tavily rate limit was exceeded`: wait for quota reset or use a higher-limit Tavily plan.
- `Private, localhost, link-local, and reserved network URLs are blocked`: use a public URL or deliberately set `WEB_RESEARCH_ALLOW_PRIVATE_NETWORK=true` for local-only testing.
- Empty catalog: verify the server URL is reachable from the `app` container, not only from the host.

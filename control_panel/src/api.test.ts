import { afterEach, describe, expect, it, vi } from 'vitest'

import { api } from './api'

function jsonResponse(body: unknown, status: number): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

describe('control-panel api error guidance', () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('maps stale backend route 404s to restart guidance', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(jsonResponse({ detail: 'Not Found' }, 404))

    await expect(api.getOverview('token')).rejects.toMatchObject({
      message: 'Connected backend does not expose control-panel routes. Restart the API from this repo.',
      status: 404,
    })
  })

  it('maps missing architecture routes to stale backend guidance', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(jsonResponse({ detail: 'Not Found' }, 404))

    await expect(api.getArchitecture('token')).rejects.toMatchObject({
      message: 'Connected backend is stale and architecture routes are unavailable. Restart the API from this repo.',
      status: 404,
    })
  })

  it('maps missing token 503s to .env guidance', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      jsonResponse({ detail: 'Control panel admin token is not configured.' }, 503),
    )

    await expect(api.getOverview('token')).rejects.toMatchObject({
      message: 'Set CONTROL_PANEL_ADMIN_TOKEN in .env and restart the API.',
      status: 503,
    })
  })

  it('maps invalid token 401s to re-entry guidance', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(jsonResponse({ detail: 'Invalid admin token.' }, 401))

    await expect(api.getOverview('token')).rejects.toMatchObject({
      message: 'The control-panel token does not match the backend. Re-enter the token or restart with the updated .env.',
      status: 401,
    })
  })

  it('falls back to openapi compatibility when the capabilities endpoint is missing', async () => {
    vi.spyOn(globalThis, 'fetch').mockImplementation(async input => {
      const url = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url
      const path = new URL(url, 'http://test.local').pathname
      if (path === '/v1/admin/capabilities') {
        return jsonResponse({ detail: 'Not Found' }, 404)
      }
      if (path === '/openapi.json') {
        return jsonResponse({
          paths: {
            '/v1/admin/overview': {},
            '/v1/admin/config/schema': {},
            '/v1/admin/config/effective': {},
            '/v1/admin/agents': {},
            '/v1/admin/prompts': {},
            '/v1/admin/collections': {},
            '/v1/admin/graphs': {},
            '/v1/admin/graphs/{graph_id}': {},
            '/v1/admin/access/principals': {},
            '/v1/admin/access/roles': {},
            '/v1/admin/access/effective-access': {},
            '/v1/admin/mcp/connections': {},
            '/v1/admin/operations': {},
            '/v1/admin/services/reset-full': {},
            '/v1/skills': {},
            '/v1/skills/build-draft': {},
          },
        }, 200)
      }
      return jsonResponse({ detail: 'Not Found' }, 404)
    })

    const result = await api.inspectCompatibility('token')

    expect(result.source).toBe('openapi')
    expect(result.capabilities.sections.dashboard.supported).toBe(true)
    expect(result.capabilities.sections.architecture.supported).toBe(false)
    expect(result.capabilities.sections.architecture.missing_routes).toEqual([
      '/v1/admin/architecture',
      '/v1/admin/architecture/activity',
    ])
  })

  it('maps missing Ollama model errors to model guidance', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      jsonResponse({ detail: 'Model `nemotron-cascade-2:30b` not found in Ollama.' }, 503),
    )

    await expect(api.getOverview('token')).rejects.toMatchObject({
      message: 'The configured Ollama model is missing. Pull the model or update the OLLAMA_* settings in .env, then restart the API.',
      status: 503,
    })
  })

  it('passes through unrelated backend messages', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(jsonResponse({ detail: 'Document not found.' }, 404))

    await expect(api.getCollectionDocument('token', 'default', 'doc-1')).rejects.toMatchObject({
      message: 'Document not found.',
      status: 404,
    })
  })

  it('preserves structured dependency errors for skill status actions', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      jsonResponse(
        {
          detail: {
            message: 'Cannot deactivate this skill because active dependents would break.',
            dependency_validation: {
              skill_id: 'skill-a',
              blocked_dependents: [{ skill_family_id: 'skill-b', name: 'Dependent Skill' }],
            },
          },
        },
        409,
      ),
    )

    await expect(api.deactivateSkill('token', 'skill-a')).rejects.toMatchObject({
      message: 'Cannot deactivate this skill because active dependents would break.',
      status: 409,
      data: {
        detail: {
          dependency_validation: {
            skill_id: 'skill-a',
          },
        },
      },
    })
  })

  it('updates agent skill assignments through the focused admin route', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      jsonResponse({ saved: true, pending_reload: true }, 200),
    )

    await api.updateAgentSkills('token', 'general', ['skill-family'])

    expect(fetchSpy).toHaveBeenCalledWith('/v1/admin/agents/general/skills', expect.objectContaining({
      method: 'PUT',
      body: JSON.stringify({ preload_skill_packs: ['skill-family'] }),
    }))
  })

  it('posts skill auto-builder drafts through the focused route', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      jsonResponse({ object: 'skill.build_draft', draft: { body_markdown: '# Skill' } }, 200),
    )

    await api.buildSkillDraft('token', {
      context: 'Build a routing workflow.',
      examples: '- Review route choice.',
      agent_scope: 'general',
    })

    expect(fetchSpy).toHaveBeenCalledWith('/v1/skills/build-draft', expect.objectContaining({
      method: 'POST',
      body: JSON.stringify({
        context: 'Build a routing workflow.',
        examples: '- Review route choice.',
        agent_scope: 'general',
      }),
    }))
  })

  it('posts the full service reset confirmation and engine', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      jsonResponse({
        status: 'started',
        run_id: 'docker-test',
        engine: 'docker',
        started_at: '2026-04-08T10:07:00Z',
        log_path: '/tmp/reset.log',
        commands: ['docker compose build --no-cache app app-bootstrap openwebui'],
      }, 200),
    )

    await api.resetServiceFull('token', 'docker')

    expect(fetchSpy).toHaveBeenCalledWith('/v1/admin/services/reset-full', expect.objectContaining({
      method: 'POST',
      body: JSON.stringify({ engine: 'docker', confirmation: 'reset-service-full' }),
    }))
  })
})

import { describe, expect, it } from 'vitest'

import { parseSectionFromPath, sectionToPath } from './navigation'

describe('control-panel navigation routes', () => {
  it('parses grouped admin and workspace routes', () => {
    expect(parseSectionFromPath('/admin/overview')).toBe('dashboard')
    expect(parseSectionFromPath('/admin/settings')).toBe('config')
    expect(parseSectionFromPath('/admin/access')).toBe('access')
    expect(parseSectionFromPath('/admin/architecture')).toBe('architecture')
    expect(parseSectionFromPath('/admin/operations')).toBe('operations')
    expect(parseSectionFromPath('/workspace/agents')).toBe('agents')
    expect(parseSectionFromPath('/workspace/knowledge')).toBe('collections')
    expect(parseSectionFromPath('/workspace/graphs')).toBe('graphs')
    expect(parseSectionFromPath('/workspace/prompts')).toBe('prompts')
    expect(parseSectionFromPath('/workspace/skills')).toBe('skills')
    expect(parseSectionFromPath('/workspace/tools')).toBe('mcp')
  })

  it('keeps legacy section routes accepted', () => {
    expect(parseSectionFromPath('/dashboard')).toBe('dashboard')
    expect(parseSectionFromPath('/config')).toBe('config')
    expect(parseSectionFromPath('/collections')).toBe('collections')
    expect(parseSectionFromPath('/mcp')).toBe('mcp')
    expect(parseSectionFromPath('/agents')).toBe('agents')
    expect(parseSectionFromPath('/architecture')).toBe('architecture')
  })

  it('emits canonical grouped routes for navigation', () => {
    expect(sectionToPath('dashboard')).toBe('/admin/overview')
    expect(sectionToPath('config')).toBe('/admin/settings')
    expect(sectionToPath('collections')).toBe('/workspace/knowledge')
    expect(sectionToPath('mcp')).toBe('/workspace/tools')
  })
})

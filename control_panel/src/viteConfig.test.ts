import { describe, expect, it } from 'vitest'

import { getBackendTarget } from './backendTarget'

describe('control-panel vite backend target', () => {
  it('defaults to localhost when no backend host override is provided', () => {
    expect(getBackendTarget({})).toBe('http://127.0.0.1:18000')
    expect(getBackendTarget({ APP_API_PORT: '8020' })).toBe('http://127.0.0.1:8020')
  })

  it('honors APP_API_HOST when the control panel runs in docker', () => {
    expect(getBackendTarget({ APP_API_HOST: 'app', APP_API_PORT: '8000' })).toBe('http://app:8000')
  })
})

import react from '@vitejs/plugin-react'
import { loadEnv } from 'vite'
import { defineConfig } from 'vitest/config'
import { getBackendTarget } from './src/backendTarget'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, '.', '')
  const backendTarget = getBackendTarget(env)

  return {
    base: mode === 'production' ? '/control-panel/' : '/',
    plugins: [react()],
    server: {
      port: 4174,
      proxy: {
        '/v1': { target: backendTarget, changeOrigin: true },
        '/health': { target: backendTarget, changeOrigin: true },
      },
    },
    test: {
      environment: 'jsdom',
      setupFiles: './src/test/setup.ts',
    },
  }
})

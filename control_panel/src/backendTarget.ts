export function getBackendTarget(env: Record<string, string | undefined>): string {
  const host = env.APP_API_HOST || '127.0.0.1'
  const port = env.APP_API_PORT || '18000'
  return `http://${host}:${port}`
}

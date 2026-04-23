export type StatusTone = 'neutral' | 'ok' | 'warning' | 'danger' | 'accent' | 'info'

export interface StatusCopy {
  label: string
  tone: StatusTone
  help?: string
}

const STATUS_COPY: Record<string, StatusCopy> = {
  ok: { label: 'Healthy', tone: 'ok' },
  active: { label: 'Active', tone: 'ok' },
  ready: { label: 'Ready', tone: 'ok' },
  success: { label: 'Succeeded', tone: 'ok' },
  configured: { label: 'Configured', tone: 'ok' },
  completed: { label: 'Completed', tone: 'ok' },
  succeeded: { label: 'Succeeded', tone: 'ok' },

  pending: { label: 'Pending', tone: 'warning' },
  warning: { label: 'Warning', tone: 'warning' },
  partial: { label: 'Partial', tone: 'warning' },
  runtime_swap: { label: 'Runtime swap', tone: 'warning', help: 'Change applied live without a restart.' },
  insufficient_data: { label: 'Insufficient data', tone: 'warning', help: 'Not enough signal yet to report a state.' },
  healthy_with_overrides: { label: 'Healthy (overrides)', tone: 'warning', help: 'Runs, but with values different from the published defaults.' },

  failed: { label: 'Failed', tone: 'danger' },
  error: { label: 'Error', tone: 'danger' },
  archived: { label: 'Archived', tone: 'danger' },
  disabled: { label: 'Disabled', tone: 'danger' },
  blocked: { label: 'Blocked', tone: 'danger' },
  flagged: { label: 'Flagged', tone: 'danger' },
  duplicate: { label: 'Duplicate', tone: 'danger' },

  live: { label: 'Live', tone: 'accent' },
  overlay: { label: 'Overlay', tone: 'accent' },
  'overlay active': { label: 'Overlay active', tone: 'accent' },

  idle: { label: 'Idle', tone: 'neutral' },
  draft: { label: 'Draft', tone: 'neutral' },
  unknown: { label: 'Unknown', tone: 'neutral' },

  scheduler_pending: { label: 'Queued', tone: 'info', help: 'Queued, waiting for a worker to pick it up.' },
  running_validation: { label: 'Validating\u2026', tone: 'info', help: 'Running pre-flight validation before execution.' },
  running: { label: 'Running\u2026', tone: 'info' },
  in_progress: { label: 'In progress', tone: 'info' },
  queued: { label: 'Queued', tone: 'info' },
  scheduled: { label: 'Scheduled', tone: 'info' },
  cancelled: { label: 'Cancelled', tone: 'neutral' },
  canceled: { label: 'Cancelled', tone: 'neutral' },

  live_config: { label: 'Live config', tone: 'ok', help: 'Change takes effect on save without a restart.' },
  startup_only: { label: 'Startup only', tone: 'warning', help: 'Requires a service restart before it takes effect.' },

  secret: { label: 'Secret', tone: 'neutral', help: 'Value is masked in the UI and redacted from audit logs.' },
  readonly: { label: 'Read-only', tone: 'neutral', help: 'Must be set via env var or host config.' },
  drafted: { label: 'Drafted', tone: 'warning', help: 'Change staged locally but not yet applied.' },
}

function normalize(value: unknown): string {
  if (value == null) return 'unknown'
  return String(value).trim().toLowerCase()
}

export function statusCopy(value: unknown): StatusCopy {
  const key = normalize(value)
  if (STATUS_COPY[key]) return STATUS_COPY[key]
  const label = key
    .replace(/[_-]+/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase())
  return { label: label || 'Unknown', tone: 'neutral' }
}

export function statusLabel(value: unknown): string {
  return statusCopy(value).label
}

export function statusTone(value: unknown): StatusTone {
  return statusCopy(value).tone
}

export function statusHelp(value: unknown): string | undefined {
  return statusCopy(value).help
}

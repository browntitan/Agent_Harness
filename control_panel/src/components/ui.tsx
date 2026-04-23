import type {
  ButtonHTMLAttributes,
  InputHTMLAttributes,
  ReactNode,
  SelectHTMLAttributes,
  TextareaHTMLAttributes,
} from 'react'
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
} from 'react'

function cx(...parts: Array<string | false | null | undefined>): string {
  return parts.filter(Boolean).join(' ')
}

export function AppShell(props: {
  sidebar: ReactNode
  header: ReactNode
  toolbar?: ReactNode
  children: ReactNode
  className?: string
}) {
  return (
    <div className={cx('app-shell', props.className)}>
      <div className="app-topbar">
        <div className="app-topbar-inner">
          {props.header}
          {props.toolbar && <div className="app-toolbar">{props.toolbar}</div>}
        </div>
      </div>
      <div className="app-layout">
        <aside className="app-sidebar">{props.sidebar}</aside>
        <main className="app-main">
          <div className="app-content">{props.children}</div>
        </main>
      </div>
    </div>
  )
}

type SidebarItem = {
  id: string
  label: string
  description?: string
  icon?: ReactNode
  badge?: ReactNode
  warning?: boolean
}

export function SidebarNav(props: {
  brand: ReactNode
  items: SidebarItem[]
  active: string
  onSelect: (id: string) => void
  footer?: ReactNode
}) {
  return (
    <div className="sidebar-frame">
      <div className="sidebar-brand">{props.brand}</div>
      <nav className="sidebar-nav" aria-label="Primary">
        {props.items.map(item => (
          <button
            key={item.id}
            type="button"
            className={cx('nav-tile', item.warning && 'nav-tile-warning', item.id === props.active && 'nav-tile-active')}
            aria-current={item.id === props.active ? 'page' : undefined}
            onClick={() => props.onSelect(item.id)}
          >
            <span className="nav-indicator" aria-hidden="true" />
            <span className="nav-icon" aria-hidden="true">{item.icon}</span>
            <span className="nav-copy">
              <span className="nav-label">{item.label}</span>
              {item.description && <span className="nav-description">{item.description}</span>}
            </span>
            {item.badge && <span className="nav-badge">{item.badge}</span>}
          </button>
        ))}
      </nav>
      {props.footer && <div className="sidebar-footer">{props.footer}</div>}
    </div>
  )
}

export function SectionHeader(props: {
  eyebrow?: string
  title: string
  description?: string
  actions?: ReactNode
}) {
  return (
    <header className="section-header">
      <div className="section-copy">
        {props.eyebrow && <span className="section-eyebrow">{props.eyebrow}</span>}
        <div>
          <h2>{props.title}</h2>
          {props.description && <p>{props.description}</p>}
        </div>
      </div>
      {props.actions && <div className="section-actions">{props.actions}</div>}
    </header>
  )
}

export function SurfaceCard(props: {
  title?: string
  subtitle?: string
  actions?: ReactNode
  className?: string
  bodyClassName?: string
  children: ReactNode
}) {
  return (
    <section className={cx('surface-card', props.className)}>
      {(props.title || props.actions || props.subtitle) && (
        <div className="surface-head">
          <div>
            {props.title && <h3>{props.title}</h3>}
            {props.subtitle && <p>{props.subtitle}</p>}
          </div>
          {props.actions && <div className="surface-actions">{props.actions}</div>}
        </div>
      )}
      <div className={cx('surface-body', props.bodyClassName)}>{props.children}</div>
    </section>
  )
}

export function CollapsibleSurfaceCard(props: {
  title?: string
  subtitle?: string
  actions?: ReactNode
  className?: string
  bodyClassName?: string
  open: boolean
  onToggle: () => void
  children: ReactNode
}) {
  const panelId = useId()

  return (
    <section className={cx('surface-card', 'collapsible-surface-card', !props.open && 'surface-card-collapsed', props.className)}>
      {(props.title || props.actions || props.subtitle) && (
        <div className="surface-head">
          <div>
            {props.title && <h3>{props.title}</h3>}
            {props.subtitle && <p>{props.subtitle}</p>}
          </div>
          <div className="surface-actions">
            {props.actions}
            <button
              type="button"
              className="collapsible-toggle"
              aria-expanded={props.open}
              aria-controls={panelId}
              onClick={props.onToggle}
            >
              {props.open ? 'Collapse' : 'Expand'}
            </button>
          </div>
        </div>
      )}
      <div
        id={panelId}
        className={cx('surface-body', 'collapsible-body', props.bodyClassName)}
        hidden={!props.open}
      >
        {props.children}
      </div>
    </section>
  )
}

export function StatCard(props: {
  label: string
  value: string | number
  caption?: string
  tone?: 'default' | 'accent' | 'warning'
}) {
  return (
    <div className={cx('stat-card', props.tone && `stat-card-${props.tone}`)}>
      <span className="stat-label">{props.label}</span>
      <strong className="stat-value">{props.value}</strong>
      {props.caption && <span className="stat-caption">{props.caption}</span>}
    </div>
  )
}

export function StatusBadge(props: {
  tone?: 'neutral' | 'ok' | 'warning' | 'danger' | 'accent' | 'info'
  iconless?: boolean
  children: ReactNode
}) {
  return (
    <span className={cx('status-badge', props.tone && `status-${props.tone}`)}>
      {!props.iconless && <span aria-hidden="true" className="status-badge-dot">●</span>}
      <span className="status-badge-label">{props.children}</span>
    </span>
  )
}

export function ActionBar(props: {
  children: ReactNode
  sticky?: boolean
}) {
  return <div className={cx('action-bar', props.sticky && 'action-bar-sticky')}>{props.children}</div>
}

type ActionButtonTone = 'primary' | 'secondary' | 'ghost' | 'destructive'

export function ActionButton(
  props: ButtonHTMLAttributes<HTMLButtonElement> & {
    tone?: ActionButtonTone
  },
) {
  const { className, tone = 'secondary', type = 'button', ...rest } = props
  return <button type={type} className={cx('action-btn', `action-btn-${tone}`, className)} {...rest} />
}

type SectionTab = {
  id: string
  label: string
}

export function SectionTabs(props: {
  tabs: SectionTab[]
  active: string
  onChange: (id: string) => void
  ariaLabel: string
  className?: string
}) {
  return (
    <div className={cx('section-switcher', props.className)} role="tablist" aria-label={props.ariaLabel}>
      {props.tabs.map(tab => (
        <button
          key={tab.id}
          type="button"
          role="tab"
          aria-selected={props.active === tab.id}
          className={props.active === tab.id ? 'section-switch-btn section-switch-btn-active' : 'section-switch-btn'}
          onClick={() => props.onChange(tab.id)}
        >
          {tab.label}
        </button>
      ))}
    </div>
  )
}

export function EntityList<T>(props: {
  title?: string
  items: T[]
  selectedKey?: string
  getKey: (item: T) => string
  getLabel: (item: T) => string
  getDescription?: (item: T) => ReactNode
  getMeta?: (item: T) => ReactNode
  emptyText?: string
  onSelect: (item: T) => void
  variant?: 'default' | 'rail'
}) {
  if (props.items.length === 0) {
    return <EmptyState title={props.title ?? 'Nothing here yet'} body={props.emptyText ?? 'No items available.'} />
  }

  return (
    <div className="entity-list">
      {props.items.map(item => {
        const key = props.getKey(item)
        const selected = key === props.selectedKey
        return (
          <button
            key={key}
            type="button"
            className={cx('entity-button', props.variant === 'rail' && 'entity-button-rail', selected && 'entity-button-active')}
            onClick={() => props.onSelect(item)}
          >
            <span className={cx(props.variant === 'rail' && 'entity-rail-main')}>
              <span className="entity-label">{props.getLabel(item)}</span>
              {props.getDescription && <span className="entity-secondary">{props.getDescription(item)}</span>}
            </span>
            {props.getMeta && <span className={cx('entity-meta', props.variant === 'rail' && 'entity-rail-meta')}>{props.getMeta(item)}</span>}
          </button>
        )
      })}
    </div>
  )
}

export function EmptyState(props: {
  title: string
  body?: ReactNode
  icon?: ReactNode
  action?: ReactNode
  tone?: 'neutral' | 'info'
}) {
  return (
    <div className={cx('empty-state', props.tone === 'info' && 'empty-state-info')}>
      {props.icon && <div className="empty-state-icon" aria-hidden="true">{props.icon}</div>}
      <div className="empty-state-copy">
        <h4>{props.title}</h4>
        {props.body && <p>{props.body}</p>}
      </div>
      {props.action && <div className="empty-state-action">{props.action}</div>}
    </div>
  )
}

export function JsonInspector(props: {
  label?: string
  value: unknown
  defaultOpen?: boolean
}) {
  return (
    <details className="json-inspector" open={props.defaultOpen}>
      <summary>{props.label ?? 'Technical details'}</summary>
      <pre>{JSON.stringify(props.value, null, 2)}</pre>
    </details>
  )
}

type DetailTab = {
  id: string
  label: string
  content: ReactNode
}

export function DetailTabs(props: {
  tabs: DetailTab[]
}) {
  const first = props.tabs[0]?.id ?? ''
  const [active, setActive] = useState(first)
  const current = props.tabs.find(tab => tab.id === active) ?? props.tabs[0]

  if (!current) return null

  return (
    <div className="detail-tabs">
      <div className="tab-row" role="tablist" aria-label="Details">
        {props.tabs.map(tab => (
          <button
            key={tab.id}
            type="button"
            role="tab"
            aria-selected={tab.id === current.id}
            className={cx('tab-pill', tab.id === current.id && 'tab-pill-active')}
            onClick={() => setActive(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </div>
      <div className="tab-panel">{current.content}</div>
    </div>
  )
}

export function SectionIcon(props: {
  kind: 'dashboard' | 'architecture' | 'config' | 'agents' | 'prompts' | 'collections' | 'graphs' | 'skills' | 'access' | 'mcp' | 'operations'
}) {
  const paths: Record<string, string> = {
    dashboard: 'M4 5.5A1.5 1.5 0 0 1 5.5 4h4A1.5 1.5 0 0 1 11 5.5v4A1.5 1.5 0 0 1 9.5 11h-4A1.5 1.5 0 0 1 4 9.5zm9 0A1.5 1.5 0 0 1 14.5 4h4A1.5 1.5 0 0 1 20 5.5v2A1.5 1.5 0 0 1 18.5 9h-4A1.5 1.5 0 0 1 13 7.5zm0 7A1.5 1.5 0 0 1 14.5 11h4A1.5 1.5 0 0 1 20 12.5v6a1.5 1.5 0 0 1-1.5 1.5h-4a1.5 1.5 0 0 1-1.5-1.5zm-9 0A1.5 1.5 0 0 1 5.5 11h4A1.5 1.5 0 0 1 11 12.5v2A1.5 1.5 0 0 1 9.5 16h-4A1.5 1.5 0 0 1 4 14.5z',
    architecture: 'M4 6.5A2.5 2.5 0 0 1 6.5 4h3A2.5 2.5 0 0 1 12 6.5v1.25h1.75A2.25 2.25 0 0 1 16 10v.75h1.5A2.5 2.5 0 0 1 20 13.25v2.25A2.5 2.5 0 0 1 17.5 18h-3A2.5 2.5 0 0 1 12 15.5v-.75h-1.75A2.25 2.25 0 0 1 8 12.5v-.75H6.5A2.5 2.5 0 0 1 4 9.25zm2.5-1A1.5 1.5 0 0 0 5 7v2.25a1.5 1.5 0 0 0 1.5 1.5H8V10A2.25 2.25 0 0 1 10.25 7.75H11V6.5A1.5 1.5 0 0 0 9.5 5zm3.75 3.25A1.25 1.25 0 0 0 9 10v2.5c0 .69.56 1.25 1.25 1.25H12V12a2.5 2.5 0 0 1 2.5-2.5H15V10c0-.69-.56-1.25-1.25-1.25zm4.25 1.75A1.5 1.5 0 0 0 13 12v3.5a1.5 1.5 0 0 0 1.5 1.5h3a1.5 1.5 0 0 0 1.5-1.5v-2.25A1.5 1.5 0 0 0 17.5 11z',
    config: 'M12 4.75a1 1 0 0 1 1 1v.61a5.77 5.77 0 0 1 1.69.7l.43-.43a1 1 0 0 1 1.41 0l.72.71a1 1 0 0 1 0 1.42l-.42.43c.3.54.54 1.1.7 1.68H19a1 1 0 0 1 1 1v1.01a1 1 0 0 1-1 1h-.61a5.8 5.8 0 0 1-.7 1.68l.42.43a1 1 0 0 1 0 1.42l-.72.71a1 1 0 0 1-1.41 0l-.43-.43a5.75 5.75 0 0 1-1.69.7V18a1 1 0 0 1-1 1h-1.01a1 1 0 0 1-1-1v-.61a5.75 5.75 0 0 1-1.69-.7l-.43.43a1 1 0 0 1-1.41 0l-.72-.71a1 1 0 0 1 0-1.42l.42-.43a5.8 5.8 0 0 1-.7-1.68H5a1 1 0 0 1-1-1V12a1 1 0 0 1 1-1h.61c.16-.58.4-1.14.7-1.68l-.42-.43a1 1 0 0 1 0-1.42l.72-.71a1 1 0 0 1 1.41 0l.43.43a5.77 5.77 0 0 1 1.69-.7v-.61a1 1 0 0 1 1-1zm0 4.25a3 3 0 1 0 0 6 3 3 0 0 0 0-6',
    agents: 'M7.5 7.25a2.75 2.75 0 1 1 0 5.5 2.75 2.75 0 0 1 0-5.5m9 1a2.25 2.25 0 1 1 0 4.5 2.25 2.25 0 0 1 0-4.5M3.75 18a3.75 3.75 0 0 1 7.5 0v.5H3.75zm8.75.5a4 4 0 0 1 2.6-3.74 3.7 3.7 0 0 1 5.15 3.24z',
    prompts: 'M6.5 4h11A2.5 2.5 0 0 1 20 6.5v11a2.5 2.5 0 0 1-2.5 2.5h-11A2.5 2.5 0 0 1 4 17.5v-11A2.5 2.5 0 0 1 6.5 4m2 4a.75.75 0 0 0 0 1.5h7a.75.75 0 0 0 0-1.5zm0 4a.75.75 0 0 0 0 1.5h7a.75.75 0 0 0 0-1.5zm0 4a.75.75 0 0 0 0 1.5H13a.75.75 0 0 0 0-1.5z',
    collections: 'M5.5 5h5.38a2 2 0 0 1 1.41.59l.71.7a2 2 0 0 0 1.42.59h4.08A1.5 1.5 0 0 1 20 8.38v8.12A2.5 2.5 0 0 1 17.5 19h-12A2.5 2.5 0 0 1 3 16.5v-9A2.5 2.5 0 0 1 5.5 5',
    graphs: 'M5.75 6.5A1.75 1.75 0 0 1 7.5 4.75h2A1.75 1.75 0 0 1 11.25 6.5v1.25h1.5A1.75 1.75 0 0 1 14.5 9.5V11h2A1.75 1.75 0 0 1 18.25 12.75v2A1.75 1.75 0 0 1 16.5 16.5h-2v1A1.75 1.75 0 0 1 12.75 19.25h-2A1.75 1.75 0 0 1 9 17.5v-1H7.5a1.75 1.75 0 0 1-1.75-1.75v-2A1.75 1.75 0 0 1 7.5 11h1.5V9.5A1.75 1.75 0 0 1 10.75 7.75h.5V6.5A.75.75 0 0 0 10.5 5.75h-2a.75.75 0 0 0-.75.75v1.25H5.75zm5 2.25a.75.75 0 0 0-.75.75V11h3V9.5a.75.75 0 0 0-.75-.75zm-3.25 3.75a.75.75 0 0 0-.75.75v2c0 .41.34.75.75.75H9v-3.5zm3 4.25a.75.75 0 0 0 .75.75h2a.75.75 0 0 0 .75-.75V12h-3.5zm4.5-4.25V16h2a.75.75 0 0 0 .75-.75v-2a.75.75 0 0 0-.75-.75z',
    skills: 'M12.7 4.3a2 2 0 0 0-2.83 0L5.4 8.77a2 2 0 0 0 0 2.83l1.41 1.41 6.3-6.3zm-5.18 9.77 3.53 3.53a2 2 0 0 0 2.83 0l4.47-4.47a2 2 0 0 0 0-2.83l-1.41-1.41zm-1.06 1.06L4 17.59V20h2.41l2.46-2.46z',
    access: 'M12 4a3 3 0 0 1 3 3v1h1.5A2.5 2.5 0 0 1 19 10.5v5A2.5 2.5 0 0 1 16.5 18H15v1a1 1 0 0 1-1 1H10a1 1 0 0 1-1-1v-1H7.5A2.5 2.5 0 0 1 5 15.5v-5A2.5 2.5 0 0 1 7.5 8H9V7a3 3 0 0 1 3-3m-1 8h2V7a1 1 0 1 0-2 0zm-3.5-2a.5.5 0 0 0-.5.5v5c0 .28.22.5.5.5h9a.5.5 0 0 0 .5-.5v-5a.5.5 0 0 0-.5-.5z',
    mcp: 'M7 5.5A2.5 2.5 0 0 1 9.5 3h5A2.5 2.5 0 0 1 17 5.5v2A2.5 2.5 0 0 1 14.5 10H13v4h1.5A2.5 2.5 0 0 1 17 16.5v2A2.5 2.5 0 0 1 14.5 21h-5A2.5 2.5 0 0 1 7 18.5v-2A2.5 2.5 0 0 1 9.5 14H11v-4H9.5A2.5 2.5 0 0 1 7 7.5zm2.5-.5A.5.5 0 0 0 9 5.5v2a.5.5 0 0 0 .5.5h5a.5.5 0 0 0 .5-.5v-2a.5.5 0 0 0-.5-.5zm0 11a.5.5 0 0 0-.5.5v2a.5.5 0 0 0 .5.5h5a.5.5 0 0 0 .5-.5v-2a.5.5 0 0 0-.5-.5z',
    operations: 'M12 5c.97 0 1.75.78 1.75 1.75V11H18a1 1 0 0 1 .7 1.71l-5.3 5.29a1 1 0 0 1-1.4 0L6.7 12.7A1 1 0 0 1 7.4 11h4.35V6.75C11.75 5.78 12.53 5 13.5 5z',
  }

  return (
    <svg className="section-icon" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
      <path d={paths[props.kind]} />
    </svg>
  )
}

// ---------------------------------------------------------------------------
// Form primitives
// ---------------------------------------------------------------------------

type FieldContextValue = {
  id: string
  describedBy: string | undefined
  invalid: boolean
  required: boolean
}

const FieldContext = createContext<FieldContextValue | null>(null)

export function FormField(props: {
  label: ReactNode
  hint?: ReactNode
  error?: ReactNode
  required?: boolean
  htmlFor?: string
  children: ReactNode
  className?: string
}) {
  const autoId = useId()
  const id = props.htmlFor ?? autoId
  const hintId = props.hint ? `${id}-hint` : undefined
  const errorId = props.error ? `${id}-error` : undefined
  const describedBy = [hintId, errorId].filter(Boolean).join(' ') || undefined
  const context: FieldContextValue = {
    id,
    describedBy,
    invalid: Boolean(props.error),
    required: Boolean(props.required),
  }

  return (
    <FieldContext.Provider value={context}>
      <div className={cx('form-field', context.invalid && 'form-field-invalid', props.className)}>
        <label htmlFor={id} className="form-field-label">
          <span>{props.label}</span>
          {props.required && <span className="form-field-required" aria-hidden="true">*</span>}
        </label>
        {props.children}
        {props.hint && !props.error && (
          <p id={hintId} className="form-field-hint">{props.hint}</p>
        )}
        {props.error && (
          <p id={errorId} className="form-field-error" role="alert">{props.error}</p>
        )}
      </div>
    </FieldContext.Provider>
  )
}

function useFieldContext() {
  return useContext(FieldContext)
}

type InputSize = 'sm' | 'md' | 'lg'

export function Input(
  props: Omit<InputHTMLAttributes<HTMLInputElement>, 'size'> & { size?: InputSize },
) {
  const { className, size = 'md', id, ...rest } = props
  const field = useFieldContext()
  return (
    <input
      id={id ?? field?.id}
      aria-invalid={field?.invalid || undefined}
      aria-describedby={field?.describedBy}
      aria-required={field?.required || undefined}
      className={cx('ctrl', 'ctrl-input', `ctrl-${size}`, className)}
      {...rest}
    />
  )
}

export function Textarea(props: TextareaHTMLAttributes<HTMLTextAreaElement>) {
  const { className, id, ...rest } = props
  const field = useFieldContext()
  return (
    <textarea
      id={id ?? field?.id}
      aria-invalid={field?.invalid || undefined}
      aria-describedby={field?.describedBy}
      aria-required={field?.required || undefined}
      className={cx('ctrl', 'ctrl-textarea', className)}
      {...rest}
    />
  )
}

export function Select(
  props: Omit<SelectHTMLAttributes<HTMLSelectElement>, 'size'> & {
    size?: InputSize
    placeholder?: string
    options?: Array<{ value: string; label: string; disabled?: boolean }>
  },
) {
  const { className, size = 'md', id, options, placeholder, children, value, ...rest } = props
  const field = useFieldContext()
  return (
    <select
      id={id ?? field?.id}
      aria-invalid={field?.invalid || undefined}
      aria-describedby={field?.describedBy}
      aria-required={field?.required || undefined}
      className={cx('ctrl', 'ctrl-select', `ctrl-${size}`, className)}
      value={value}
      {...rest}
    >
      {placeholder !== undefined && (
        <option value="" disabled={value !== ''}>
          {placeholder}
        </option>
      )}
      {options
        ? options.map(opt => (
            <option key={opt.value} value={opt.value} disabled={opt.disabled}>
              {opt.label}
            </option>
          ))
        : children}
    </select>
  )
}

export function Switch(props: {
  checked: boolean
  onChange: (next: boolean) => void
  label?: ReactNode
  hint?: ReactNode
  disabled?: boolean
  id?: string
  'aria-label'?: string
}) {
  const autoId = useId()
  const id = props.id ?? autoId
  const hintId = props.hint ? `${id}-hint` : undefined
  return (
    <div className={cx('switch-field', props.disabled && 'switch-field-disabled')}>
      <button
        id={id}
        type="button"
        role="switch"
        aria-checked={props.checked}
        aria-label={props['aria-label']}
        aria-describedby={hintId}
        disabled={props.disabled}
        className={cx('switch-track', props.checked && 'switch-track-on')}
        onClick={() => props.onChange(!props.checked)}
      >
        <span className="switch-thumb" aria-hidden="true" />
      </button>
      {(props.label || props.hint) && (
        <div className="switch-copy">
          {props.label && <label htmlFor={id} className="switch-label">{props.label}</label>}
          {props.hint && <span id={hintId} className="switch-hint">{props.hint}</span>}
        </div>
      )}
    </div>
  )
}

export function Checkbox(
  props: Omit<InputHTMLAttributes<HTMLInputElement>, 'type'> & { label?: ReactNode },
) {
  const { className, label, id, ...rest } = props
  const autoId = useId()
  const resolvedId = id ?? autoId
  return (
    <label htmlFor={resolvedId} className={cx('checkbox-field', className)}>
      <input id={resolvedId} type="checkbox" className="checkbox-input" {...rest} />
      {label && <span className="checkbox-label">{label}</span>}
    </label>
  )
}

// ---------------------------------------------------------------------------
// Buttons
// ---------------------------------------------------------------------------

type ButtonTone = 'primary' | 'secondary' | 'ghost' | 'destructive' | 'caution'
type ButtonSize = 'sm' | 'md' | 'lg'

export function Button(
  props: ButtonHTMLAttributes<HTMLButtonElement> & {
    tone?: ButtonTone
    size?: ButtonSize
    iconLeft?: ReactNode
    iconRight?: ReactNode
    loading?: boolean
  },
) {
  const {
    className,
    tone = 'secondary',
    size = 'md',
    type = 'button',
    iconLeft,
    iconRight,
    loading,
    disabled,
    children,
    ...rest
  } = props
  return (
    <button
      type={type}
      className={cx('action-btn', `action-btn-${tone}`, `action-btn-${size}`, loading && 'action-btn-loading', className)}
      disabled={disabled || loading}
      aria-busy={loading || undefined}
      {...rest}
    >
      {loading && <span className="action-btn-spinner" aria-hidden="true" />}
      {iconLeft && <span className="action-btn-icon" aria-hidden="true">{iconLeft}</span>}
      {children && <span className="action-btn-label">{children}</span>}
      {iconRight && <span className="action-btn-icon" aria-hidden="true">{iconRight}</span>}
    </button>
  )
}

export function IconButton(
  props: ButtonHTMLAttributes<HTMLButtonElement> & {
    tone?: ButtonTone
    size?: ButtonSize
    'aria-label': string
  },
) {
  const { className, tone = 'ghost', size = 'md', type = 'button', ...rest } = props
  return (
    <button
      type={type}
      className={cx('icon-btn', `icon-btn-${tone}`, `icon-btn-${size}`, className)}
      {...rest}
    />
  )
}

export function CopyButton(props: {
  value: string
  label?: string
  tone?: ButtonTone
  size?: ButtonSize
  className?: string
}) {
  const [copied, setCopied] = useState(false)
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  useEffect(() => () => {
    if (timerRef.current) clearTimeout(timerRef.current)
  }, [])
  const onClick = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(props.value)
      setCopied(true)
      if (timerRef.current) clearTimeout(timerRef.current)
      timerRef.current = setTimeout(() => setCopied(false), 1600)
    } catch {
      /* swallow — copy unavailable */
    }
  }, [props.value])
  return (
    <Button
      tone={props.tone ?? 'ghost'}
      size={props.size ?? 'sm'}
      className={cx('copy-btn', props.className)}
      onClick={onClick}
      aria-live="polite"
    >
      {copied ? 'Copied' : props.label ?? 'Copy'}
    </Button>
  )
}

// ---------------------------------------------------------------------------
// Overlays: Dialog + ConfirmDialog
// ---------------------------------------------------------------------------

export function Dialog(props: {
  open: boolean
  onClose: () => void
  title: ReactNode
  description?: ReactNode
  children?: ReactNode
  footer?: ReactNode
  size?: 'sm' | 'md' | 'lg'
  closeOnBackdrop?: boolean
  labelId?: string
}) {
  const autoId = useId()
  const labelId = props.labelId ?? `${autoId}-title`
  const descId = props.description ? `${autoId}-desc` : undefined
  const containerRef = useRef<HTMLDivElement | null>(null)
  const previousActiveRef = useRef<Element | null>(null)

  useEffect(() => {
    if (!props.open) return
    previousActiveRef.current = document.activeElement
    const node = containerRef.current
    if (node) {
      const focusable = node.querySelector<HTMLElement>(
        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])',
      )
      focusable?.focus()
    }
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.stopPropagation()
        props.onClose()
      }
    }
    document.addEventListener('keydown', onKey)
    const { body } = document
    const prevOverflow = body.style.overflow
    body.style.overflow = 'hidden'
    return () => {
      document.removeEventListener('keydown', onKey)
      body.style.overflow = prevOverflow
      if (previousActiveRef.current instanceof HTMLElement) {
        previousActiveRef.current.focus()
      }
    }
  }, [props.open, props.onClose])

  if (!props.open) return null

  return (
    <div
      className="dialog-backdrop"
      role="presentation"
      onMouseDown={event => {
        if (event.target === event.currentTarget && props.closeOnBackdrop !== false) props.onClose()
      }}
    >
      <div
        ref={containerRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby={labelId}
        aria-describedby={descId}
        className={cx('dialog', `dialog-${props.size ?? 'md'}`)}
      >
        <header className="dialog-head">
          <h2 id={labelId}>{props.title}</h2>
          <IconButton aria-label="Close dialog" onClick={props.onClose}>
            <svg viewBox="0 0 20 20" width="16" height="16" fill="currentColor" aria-hidden="true">
              <path d="M5.3 4.3a1 1 0 0 1 1.4 0L10 7.6l3.3-3.3a1 1 0 1 1 1.4 1.4L11.4 9l3.3 3.3a1 1 0 1 1-1.4 1.4L10 10.4l-3.3 3.3a1 1 0 1 1-1.4-1.4L8.6 9 5.3 5.7a1 1 0 0 1 0-1.4" />
            </svg>
          </IconButton>
        </header>
        {props.description && <p id={descId} className="dialog-desc">{props.description}</p>}
        {props.children && <div className="dialog-body">{props.children}</div>}
        {props.footer && <footer className="dialog-footer">{props.footer}</footer>}
      </div>
    </div>
  )
}

export function ConfirmDialog(props: {
  open: boolean
  onCancel: () => void
  onConfirm: () => void | Promise<void>
  title: ReactNode
  description?: ReactNode
  confirmLabel?: string
  cancelLabel?: string
  tone?: 'destructive' | 'caution' | 'primary'
  loading?: boolean
}) {
  const tone = props.tone ?? 'primary'
  return (
    <Dialog
      open={props.open}
      onClose={props.onCancel}
      title={props.title}
      description={props.description}
      size="sm"
      footer={
        <>
          <Button tone="ghost" onClick={props.onCancel} disabled={props.loading}>
            {props.cancelLabel ?? 'Cancel'}
          </Button>
          <Button tone={tone} onClick={() => void props.onConfirm()} loading={props.loading}>
            {props.confirmLabel ?? 'Confirm'}
          </Button>
        </>
      }
    />
  )
}

// ---------------------------------------------------------------------------
// Toast
// ---------------------------------------------------------------------------

type Toast = {
  id: string
  tone: 'info' | 'ok' | 'warn' | 'danger'
  title: ReactNode
  body?: ReactNode
  durationMs?: number
}

type ToastContextValue = {
  push: (toast: Omit<Toast, 'id'>) => string
  dismiss: (id: string) => void
}

const ToastContext = createContext<ToastContextValue | null>(null)

export function ToastProvider(props: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([])
  const timersRef = useRef(new Map<string, ReturnType<typeof setTimeout>>())

  const dismiss = useCallback((id: string) => {
    setToasts(prev => prev.filter(t => t.id !== id))
    const timer = timersRef.current.get(id)
    if (timer) {
      clearTimeout(timer)
      timersRef.current.delete(id)
    }
  }, [])

  const push = useCallback<ToastContextValue['push']>(toast => {
    const id = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
    setToasts(prev => [...prev, { ...toast, id }])
    const duration = toast.durationMs ?? 5000
    if (duration > 0) {
      const timer = setTimeout(() => dismiss(id), duration)
      timersRef.current.set(id, timer)
    }
    return id
  }, [dismiss])

  useEffect(() => () => {
    timersRef.current.forEach(clearTimeout)
    timersRef.current.clear()
  }, [])

  const value = useMemo(() => ({ push, dismiss }), [push, dismiss])

  return (
    <ToastContext.Provider value={value}>
      {props.children}
      <div className="toast-region" role="region" aria-live="polite" aria-label="Notifications">
        {toasts.map(toast => (
          <div key={toast.id} className={cx('toast', `toast-${toast.tone}`)} role="status">
            <div className="toast-copy">
              <strong className="toast-title">{toast.title}</strong>
              {toast.body && <span className="toast-body">{toast.body}</span>}
            </div>
            <IconButton aria-label="Dismiss notification" size="sm" onClick={() => dismiss(toast.id)}>
              <svg viewBox="0 0 20 20" width="14" height="14" fill="currentColor" aria-hidden="true">
                <path d="M5.3 4.3a1 1 0 0 1 1.4 0L10 7.6l3.3-3.3a1 1 0 1 1 1.4 1.4L11.4 9l3.3 3.3a1 1 0 1 1-1.4 1.4L10 10.4l-3.3 3.3a1 1 0 1 1-1.4-1.4L8.6 9 5.3 5.7a1 1 0 0 1 0-1.4" />
              </svg>
            </IconButton>
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  )
}

export function useToast(): ToastContextValue {
  const ctx = useContext(ToastContext)
  if (!ctx) throw new Error('useToast must be used within a ToastProvider')
  return ctx
}

// ---------------------------------------------------------------------------
// Content primitives
// ---------------------------------------------------------------------------

export function Tag(props: {
  tone?: 'neutral' | 'ok' | 'warn' | 'danger' | 'info' | 'accent'
  children: ReactNode
  className?: string
}) {
  return <span className={cx('tag', `tag-${props.tone ?? 'neutral'}`, props.className)}>{props.children}</span>
}

export function Kbd(props: { children: ReactNode; className?: string }) {
  return <kbd className={cx('kbd', props.className)}>{props.children}</kbd>
}

export function DescriptionList(props: {
  items: Array<{ term: ReactNode; detail: ReactNode; key?: string }>
  dense?: boolean
  className?: string
}) {
  return (
    <dl className={cx('description-list', props.dense && 'description-list-dense', props.className)}>
      {props.items.map((item, index) => (
        <div key={item.key ?? index} className="description-list-row">
          <dt>{item.term}</dt>
          <dd>{item.detail}</dd>
        </div>
      ))}
    </dl>
  )
}

export function CodeBlock(props: {
  code: string
  language?: string
  copyable?: boolean
  ariaLabel?: string
  maxHeight?: number
  className?: string
}) {
  return (
    <div className={cx('code-block', props.className)}>
      {props.copyable && (
        <div className="code-block-actions">
          <CopyButton value={props.code} />
        </div>
      )}
      <pre
        className="code-block-pre"
        aria-label={props.ariaLabel}
        data-language={props.language}
        style={props.maxHeight ? { maxHeight: props.maxHeight } : undefined}
      >
        <code>{props.code}</code>
      </pre>
    </div>
  )
}

export function Breadcrumb(props: {
  items: Array<{ label: ReactNode; onSelect?: () => void; current?: boolean }>
}) {
  return (
    <nav className="breadcrumb" aria-label="Breadcrumb">
      <ol>
        {props.items.map((item, index) => {
          const last = index === props.items.length - 1 || item.current
          return (
            <li key={index} className="breadcrumb-item">
              {last || !item.onSelect ? (
                <span aria-current={last ? 'page' : undefined}>{item.label}</span>
              ) : (
                <button type="button" className="breadcrumb-link" onClick={item.onSelect}>
                  {item.label}
                </button>
              )}
              {!last && <span className="breadcrumb-sep" aria-hidden="true">/</span>}
            </li>
          )
        })}
      </ol>
    </nav>
  )
}

export function Skeleton(props: {
  width?: number | string
  height?: number | string
  radius?: 'sm' | 'md' | 'lg' | 'pill'
  className?: string
}) {
  return (
    <span
      aria-hidden="true"
      className={cx('skeleton', props.radius && `skeleton-${props.radius}`, props.className)}
      style={{ width: props.width, height: props.height }}
    />
  )
}

export function ErrorState(props: {
  title: ReactNode
  body?: ReactNode
  action?: ReactNode
}) {
  return (
    <div className="error-state" role="alert">
      <h4>{props.title}</h4>
      {props.body && <p>{props.body}</p>}
      {props.action && <div className="error-state-action">{props.action}</div>}
    </div>
  )
}

type TooltipPlacement = 'top' | 'bottom' | 'left' | 'right'

export function Tooltip(props: {
  content: ReactNode
  placement?: TooltipPlacement
  delay?: number
  children: ReactNode
}) {
  const placement = props.placement ?? 'top'
  const delay = props.delay ?? 200
  const tooltipId = useId()
  const wrapperRef = useRef<HTMLSpanElement | null>(null)
  const tipRef = useRef<HTMLSpanElement | null>(null)
  const timerRef = useRef<number | null>(null)
  const [open, setOpen] = useState(false)
  const [coords, setCoords] = useState<{ top: number; left: number } | null>(null)

  const clearTimer = useCallback(() => {
    if (timerRef.current !== null) {
      window.clearTimeout(timerRef.current)
      timerRef.current = null
    }
  }, [])

  const schedule = useCallback(() => {
    clearTimer()
    timerRef.current = window.setTimeout(() => setOpen(true), delay)
  }, [clearTimer, delay])

  const dismiss = useCallback(() => {
    clearTimer()
    setOpen(false)
  }, [clearTimer])

  useEffect(() => () => clearTimer(), [clearTimer])

  useEffect(() => {
    if (!open) return
    const target = wrapperRef.current
    const tip = tipRef.current
    if (!target || !tip) return
    const rect = target.getBoundingClientRect()
    const tipRect = tip.getBoundingClientRect()
    const gap = 8
    let top = 0
    let left = 0
    if (placement === 'top') {
      top = rect.top - tipRect.height - gap
      left = rect.left + rect.width / 2 - tipRect.width / 2
    } else if (placement === 'bottom') {
      top = rect.bottom + gap
      left = rect.left + rect.width / 2 - tipRect.width / 2
    } else if (placement === 'left') {
      top = rect.top + rect.height / 2 - tipRect.height / 2
      left = rect.left - tipRect.width - gap
    } else {
      top = rect.top + rect.height / 2 - tipRect.height / 2
      left = rect.right + gap
    }
    const margin = 4
    const maxLeft = window.innerWidth - tipRect.width - margin
    const maxTop = window.innerHeight - tipRect.height - margin
    setCoords({
      top: Math.max(margin, Math.min(top, maxTop)),
      left: Math.max(margin, Math.min(left, maxLeft)),
    })
  }, [open, placement])

  useEffect(() => {
    if (!open) return
    function onScrollOrResize() { setOpen(false) }
    function onKey(e: KeyboardEvent) { if (e.key === 'Escape') setOpen(false) }
    window.addEventListener('scroll', onScrollOrResize, true)
    window.addEventListener('resize', onScrollOrResize)
    window.addEventListener('keydown', onKey)
    return () => {
      window.removeEventListener('scroll', onScrollOrResize, true)
      window.removeEventListener('resize', onScrollOrResize)
      window.removeEventListener('keydown', onKey)
    }
  }, [open])

  return (
    <span
      ref={wrapperRef}
      className="tooltip-wrapper"
      onMouseEnter={schedule}
      onMouseLeave={dismiss}
      onFocus={schedule}
      onBlur={dismiss}
      aria-describedby={open ? tooltipId : undefined}
    >
      {props.children}
      {open && (
        <span
          ref={tipRef}
          id={tooltipId}
          role="tooltip"
          className={cx('tooltip', `tooltip-${placement}`)}
          style={coords ? { top: coords.top, left: coords.left, visibility: 'visible' } : { visibility: 'hidden' }}
        >
          {props.content}
        </span>
      )}
    </span>
  )
}

import {
  Activity,
  Bot,
  Boxes,
  Database,
  GitBranch,
  KeyRound,
  LockKeyhole,
  Moon,
  Network,
  PanelLeftClose,
  ScrollText,
  Search,
  Settings2,
  ShieldCheck,
  Sparkles,
  Sun,
  Wrench,
} from 'lucide-react'
import type { Density } from '../theme/DensityProvider'
import type { Theme } from '../theme/ThemeProvider'
import { sectionToPath, sectionsForGroup, type Section } from '../navigation'
import { ActionButton, IconButton, Tooltip } from './ui'

function sectionNavIcon(section: Section) {
  const iconProps = { size: 15, strokeWidth: 1.9, 'aria-hidden': true }
  switch (section) {
    case 'dashboard':
      return <Boxes {...iconProps} />
    case 'config':
      return <Settings2 {...iconProps} />
    case 'access':
      return <ShieldCheck {...iconProps} />
    case 'architecture':
      return <Network {...iconProps} />
    case 'operations':
      return <Activity {...iconProps} />
    case 'agents':
      return <Bot {...iconProps} />
    case 'collections':
      return <Database {...iconProps} />
    case 'graphs':
      return <GitBranch {...iconProps} />
    case 'prompts':
      return <ScrollText {...iconProps} />
    case 'skills':
      return <Sparkles {...iconProps} />
    case 'mcp':
      return <Wrench {...iconProps} />
  }
}

export function ControlPanelTopNav(props: {
  active: Section
  onSelect: (section: Section) => void
  sectionCounts: Map<Section, number>
  unsupportedSectionIds: Section[]
  theme: Theme
  density: Density
  onOpenPalette: () => void
  onToggleTheme: () => void
  onToggleDensity: () => void
  onLock: () => void
}) {
  return (
    <div className="owui-nav-shell">
      <div className="owui-brand">
        <div className="owui-brand-mark" aria-hidden="true">
          <KeyRound size={15} strokeWidth={2} />
        </div>
        <div className="owui-brand-copy">
          <span>Agentic RAG</span>
          <strong>Control Panel</strong>
        </div>
      </div>

      <div className="owui-nav-groups">
        {(['admin', 'workspace'] as const).map(group => (
          <div className="owui-nav-group" key={group}>
            <span className="owui-nav-group-label">{group === 'admin' ? 'Admin Panel' : 'Workspace'}</span>
            <nav className="owui-nav-strip" aria-label={group === 'admin' ? 'Admin Panel' : 'Workspace'}>
              {sectionsForGroup(group).map(section => {
                const unsupported = props.unsupportedSectionIds.includes(section.id)
                const count = props.sectionCounts.get(section.id)
                return (
                  <button
                    key={section.id}
                    type="button"
                    className={[
                      'owui-nav-item',
                      props.active === section.id ? 'owui-nav-item-active' : '',
                      unsupported ? 'owui-nav-item-warning' : '',
                    ].filter(Boolean).join(' ')}
                    aria-current={props.active === section.id ? 'page' : undefined}
                    onClick={() => props.onSelect(section.id)}
                    data-route={sectionToPath(section.id)}
                  >
                    <span className="owui-nav-icon">{sectionNavIcon(section.id)}</span>
                    <span className="owui-nav-label">{section.shortLabel ?? section.label}</span>
                    {unsupported ? (
                      <span className="owui-nav-dot" aria-label="Unsupported" />
                    ) : typeof count === 'number' && count > 0 ? (
                      <span className="owui-nav-count">{count.toLocaleString()}</span>
                    ) : null}
                  </button>
                )
              })}
            </nav>
          </div>
        ))}
      </div>

      <div className="owui-global-actions">
        <Tooltip content="Open command palette">
          <IconButton aria-label="Open command palette" onClick={props.onOpenPalette} size="sm">
            <Search size={15} strokeWidth={2} />
          </IconButton>
        </Tooltip>
        <Tooltip content={props.theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}>
          <IconButton
            aria-label={props.theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
            onClick={props.onToggleTheme}
            size="sm"
          >
            {props.theme === 'dark' ? <Sun size={15} strokeWidth={2} /> : <Moon size={15} strokeWidth={2} />}
          </IconButton>
        </Tooltip>
        <Tooltip content={props.density === 'comfortable' ? 'Switch to compact density' : 'Switch to comfortable density'}>
          <IconButton
            aria-label={props.density === 'comfortable' ? 'Switch to compact density' : 'Switch to comfortable density'}
            aria-pressed={props.density === 'compact'}
            onClick={props.onToggleDensity}
            size="sm"
          >
            <PanelLeftClose size={15} strokeWidth={2} />
          </IconButton>
        </Tooltip>
        <ActionButton tone="ghost" onClick={props.onLock}>
          <span className="button-with-icon"><LockKeyhole size={14} strokeWidth={2} />Lock</span>
        </ActionButton>
      </div>
    </div>
  )
}

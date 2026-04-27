import { useMemo, useState } from 'react'
import type {
  AccessPrincipal,
  AccessRole,
  AccessRoleBinding,
  AccessRolePermission,
} from '../../types'
import { EmptyState, FilterChip, Popover, SegmentedControl, Tooltip } from '../../components/ui'

type PrincipalFilter = 'all' | 'user' | 'group' | 'service'
type ActionLevel = 'none' | 'use' | 'manage'

const RESOURCE_TYPES: Array<{ key: string; label: string }> = [
  { key: 'collection', label: 'Collections' },
  { key: 'graph', label: 'Graphs' },
  { key: 'tool', label: 'Tools' },
  { key: 'skill_family', label: 'Skills' },
]

function strongerAction(a: ActionLevel, b: ActionLevel): ActionLevel {
  if (a === 'manage' || b === 'manage') return 'manage'
  if (a === 'use' || b === 'use') return 'use'
  return 'none'
}

function glyphForLevel(level: ActionLevel): string {
  if (level === 'manage') return '●'
  if (level === 'use') return '○'
  return '·'
}

function principalLabel(p: AccessPrincipal): string {
  return p.display_name || p.email_normalized || p.principal_id
}

function principalCategory(p: AccessPrincipal): PrincipalFilter {
  const t = (p.principal_type || '').toLowerCase()
  if (t === 'user') return 'user'
  if (t === 'group') return 'group'
  return 'service'
}

interface Grant {
  level: ActionLevel
  selectors: string[]
  via: Array<{ roleName: string; selector: string; action: string }>
}

export function RbacMatrix(props: {
  principals: AccessPrincipal[]
  roles: AccessRole[]
  bindings: AccessRoleBinding[]
  permissions: AccessRolePermission[]
}) {
  const [filter, setFilter] = useState<PrincipalFilter>('all')
  const [onlyDifferences, setOnlyDifferences] = useState(false)
  const [activeTypes, setActiveTypes] = useState<Set<string>>(() => new Set(RESOURCE_TYPES.map(r => r.key)))

  const rolesById = useMemo(() => new Map(props.roles.map(r => [r.role_id, r])), [props.roles])
  const permsByRole = useMemo(() => {
    const map = new Map<string, AccessRolePermission[]>()
    for (const perm of props.permissions) {
      const list = map.get(perm.role_id) ?? []
      list.push(perm)
      map.set(perm.role_id, list)
    }
    return map
  }, [props.permissions])

  const rows = useMemo(() => {
    const filtered = props.principals
      .filter(p => p.active !== false)
      .filter(p => filter === 'all' || principalCategory(p) === filter)

    const grid: Array<{ principal: AccessPrincipal; grants: Record<string, Grant> }> = []
    for (const principal of filtered) {
      const principalBindings = props.bindings.filter(
        b => b.principal_id === principal.principal_id && !b.disabled,
      )
      const grants: Record<string, Grant> = {}
      for (const rt of RESOURCE_TYPES) {
        grants[rt.key] = { level: 'none', selectors: [], via: [] }
      }
      for (const binding of principalBindings) {
        const role = rolesById.get(binding.role_id)
        const rolePerms = permsByRole.get(binding.role_id) ?? []
        for (const perm of rolePerms) {
          const g = grants[perm.resource_type]
          if (!g) continue
          const action = (perm.action === 'manage' ? 'manage' : 'use') as ActionLevel
          g.level = strongerAction(g.level, action)
          if (!g.selectors.includes(perm.resource_selector)) g.selectors.push(perm.resource_selector)
          g.via.push({
            roleName: role?.name ?? binding.role_id,
            selector: perm.resource_selector,
            action: perm.action,
          })
        }
      }
      grid.push({ principal, grants })
    }

    if (onlyDifferences && grid.length > 1) {
      return grid.filter(row => {
        const signature = RESOURCE_TYPES.map(rt => row.grants[rt.key].level).join('|')
        return grid.some(other => {
          if (other === row) return false
          const otherSig = RESOURCE_TYPES.map(rt => other.grants[rt.key].level).join('|')
          return otherSig !== signature
        })
      })
    }
    return grid
  }, [props.principals, props.bindings, rolesById, permsByRole, filter, onlyDifferences])

  const visibleColumns = RESOURCE_TYPES.filter(rt => activeTypes.has(rt.key))

  function toggleType(key: string) {
    setActiveTypes(prev => {
      const next = new Set(prev)
      if (next.has(key)) next.delete(key); else next.add(key)
      if (next.size === 0) return new Set(RESOURCE_TYPES.map(r => r.key))
      return next
    })
  }

  if (props.principals.length === 0) {
    return <EmptyState title="No principals to chart" body="Add a user or group to see the access matrix." />
  }

  return (
    <div className="rbac-matrix">
      <div className="rbac-matrix-toolbar">
        <SegmentedControl<PrincipalFilter>
          size="sm"
          ariaLabel="Filter principals"
          value={filter}
          onChange={setFilter}
          options={[
            { value: 'all', label: 'All' },
            { value: 'user', label: 'Users' },
            { value: 'group', label: 'Groups' },
            { value: 'service', label: 'Services' },
          ]}
        />
        <div className="rbac-matrix-chips">
          {RESOURCE_TYPES.map(rt => (
            <button
              key={rt.key}
              type="button"
              className="rbac-chip-btn"
              onClick={() => toggleType(rt.key)}
              aria-pressed={activeTypes.has(rt.key)}
            >
              <FilterChip
                label={rt.label}
                tone={activeTypes.has(rt.key) ? 'accent' : 'neutral'}
              />
            </button>
          ))}
        </div>
        <label className="rbac-matrix-toggle">
          <input
            type="checkbox"
            checked={onlyDifferences}
            onChange={e => setOnlyDifferences(e.target.checked)}
          />
          <span>Show only differences</span>
        </label>
      </div>

      <div className="rbac-matrix-legend">
        <span><span className="rbac-glyph rbac-glyph-none">·</span> None</span>
        <span><span className="rbac-glyph rbac-glyph-use">○</span> Use</span>
        <span><span className="rbac-glyph rbac-glyph-manage">●</span> Manage</span>
      </div>

      <div
        className="rbac-matrix-grid"
        role="table"
        aria-label="Role-based access matrix"
        style={{ gridTemplateColumns: `minmax(220px, 1.4fr) repeat(${visibleColumns.length}, minmax(80px, 1fr))` }}
      >
        <div className="rbac-matrix-header" role="row">
          <div className="rbac-matrix-corner" role="columnheader">Principal</div>
          {visibleColumns.map(col => (
            <div key={col.key} className="rbac-matrix-col-head" role="columnheader">
              {col.label}
            </div>
          ))}
        </div>

        {rows.length === 0 ? (
          <div className="rbac-matrix-empty" role="row">
            <EmptyState title="No matches" body="Adjust the filter or toggle more resource types to see principals." />
          </div>
        ) : rows.map(({ principal, grants }) => (
          <div key={principal.principal_id} className="rbac-matrix-row" role="row">
            <div className="rbac-matrix-row-head" role="rowheader">
              <strong>{principalLabel(principal)}</strong>
              <span className="rbac-matrix-subtitle">
                {principal.principal_type || 'service'} · {principal.email_normalized || principal.principal_id}
              </span>
            </div>
            {visibleColumns.map(col => {
              const grant = grants[col.key]
              const glyph = glyphForLevel(grant.level)
              const tooltipLines = grant.via.length === 0
                ? 'No grant'
                : grant.via.slice(0, 4).map(v => `${v.action} ${col.label.toLowerCase()} "${v.selector}" via ${v.roleName}`).join('\n')
              if (grant.level === 'none') {
                return (
                  <div key={col.key} className="rbac-matrix-cell" role="cell">
                    <Tooltip content={tooltipLines}>
                      <span className={`rbac-glyph rbac-glyph-none`}>{glyph}</span>
                    </Tooltip>
                  </div>
                )
              }
              return (
                <div key={col.key} className="rbac-matrix-cell rbac-matrix-cell-has-grant" role="cell">
                  <Popover
                    ariaLabel={`${principalLabel(principal)} access to ${col.label}`}
                    trigger={({ toggle, ref }) => (
                      <button
                        type="button"
                        ref={el => ref(el)}
                        onClick={toggle}
                        className={`rbac-cell-btn rbac-cell-btn-${grant.level}`}
                        aria-label={`${principalLabel(principal)} ${grant.level} on ${col.label}`}
                      >
                        <span className={`rbac-glyph rbac-glyph-${grant.level}`}>{glyph}</span>
                      </button>
                    )}
                  >
                    <div className="rbac-cell-popover">
                      <header className="rbac-cell-popover-head">
                        <span className="section-eyebrow">{col.label}</span>
                        <strong>{principalLabel(principal)}</strong>
                      </header>
                      <ul className="rbac-cell-popover-list">
                        {grant.via.map((v, i) => (
                          <li key={i}>
                            <span className={`rbac-glyph rbac-glyph-${v.action === 'manage' ? 'manage' : 'use'}`} aria-hidden="true">
                              {v.action === 'manage' ? '●' : '○'}
                            </span>
                            <span>
                              <strong>{v.action}</strong> <code>{v.selector}</code>
                            </span>
                            <span className="rbac-cell-popover-role">via {v.roleName}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  </Popover>
                </div>
              )
            })}
          </div>
        ))}
      </div>
    </div>
  )
}

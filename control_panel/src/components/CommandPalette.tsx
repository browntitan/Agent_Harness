import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Kbd } from './ui'

export interface PaletteCommand {
  id: string
  label: string
  hint?: string
  group?: string
  keywords?: string[]
  run: () => void | Promise<void>
}

function score(command: PaletteCommand, query: string): number {
  if (!query) return 1
  const haystack = [command.label, command.group ?? '', ...(command.keywords ?? [])]
    .join(' ')
    .toLowerCase()
  const needle = query.toLowerCase().trim()
  if (!needle) return 1
  if (haystack.includes(needle)) return 2
  let ni = 0
  for (let i = 0; i < haystack.length && ni < needle.length; i++) {
    if (haystack[i] === needle[ni]) ni++
  }
  return ni === needle.length ? 1 : 0
}

export function CommandPalette(props: {
  open: boolean
  onClose: () => void
  commands: PaletteCommand[]
}) {
  const [query, setQuery] = useState('')
  const [activeIndex, setActiveIndex] = useState(0)
  const inputRef = useRef<HTMLInputElement | null>(null)
  const listRef = useRef<HTMLUListElement | null>(null)

  useEffect(() => {
    if (!props.open) return
    setQuery('')
    setActiveIndex(0)
    const frame = requestAnimationFrame(() => inputRef.current?.focus())
    return () => cancelAnimationFrame(frame)
  }, [props.open])

  const filtered = useMemo(() => {
    const scored: Array<{ cmd: PaletteCommand; s: number }> = []
    for (const cmd of props.commands) {
      const s = score(cmd, query)
      if (s > 0) scored.push({ cmd, s })
    }
    scored.sort((a, b) => b.s - a.s)
    return scored.map(x => x.cmd)
  }, [props.commands, query])

  const groups = useMemo(() => {
    const map = new Map<string, PaletteCommand[]>()
    for (const cmd of filtered) {
      const group = cmd.group ?? 'Commands'
      const list = map.get(group) ?? []
      list.push(cmd)
      map.set(group, list)
    }
    return Array.from(map.entries())
  }, [filtered])

  useEffect(() => { setActiveIndex(0) }, [query])

  const runAt = useCallback(async (idx: number) => {
    const cmd = filtered[idx]
    if (!cmd) return
    props.onClose()
    await cmd.run()
  }, [filtered, props])

  const onKey = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setActiveIndex(i => Math.min(filtered.length - 1, i + 1))
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setActiveIndex(i => Math.max(0, i - 1))
    } else if (e.key === 'Enter') {
      e.preventDefault()
      void runAt(activeIndex)
    } else if (e.key === 'Escape') {
      e.preventDefault()
      props.onClose()
    }
  }, [filtered.length, activeIndex, runAt, props])

  useEffect(() => {
    if (!listRef.current) return
    const items = listRef.current.querySelectorAll<HTMLLIElement>('[data-palette-item]')
    items[activeIndex]?.scrollIntoView({ block: 'nearest' })
  }, [activeIndex])

  if (!props.open) return null

  let runningIndex = -1

  return (
    <div className="palette-backdrop" role="presentation" onMouseDown={e => { if (e.target === e.currentTarget) props.onClose() }}>
      <div role="dialog" aria-modal="true" aria-label="Command palette" className="palette">
        <div className="palette-input-row">
          <svg viewBox="0 0 20 20" width="16" height="16" fill="currentColor" aria-hidden="true" className="palette-input-icon">
            <path d="M9 3a6 6 0 1 0 3.75 10.66l3.29 3.29a1 1 0 0 0 1.42-1.42l-3.29-3.29A6 6 0 0 0 9 3m0 2a4 4 0 1 1 0 8 4 4 0 0 1 0-8" />
          </svg>
          <input
            ref={inputRef}
            type="text"
            role="combobox"
            aria-expanded="true"
            aria-controls="palette-list"
            aria-activedescendant={`palette-item-${activeIndex}`}
            placeholder="Type a command or search…"
            value={query}
            onChange={e => setQuery(e.target.value)}
            onKeyDown={onKey}
            className="palette-input"
          />
          <Kbd>Esc</Kbd>
        </div>
        <ul ref={listRef} id="palette-list" role="listbox" className="palette-list">
          {groups.length === 0 && (
            <li className="palette-empty">No commands match "{query}"</li>
          )}
          {groups.map(([group, items]) => (
            <li key={group} className="palette-group">
              <div className="palette-group-head">{group}</div>
              <ul role="group" aria-label={group}>
                {items.map(cmd => {
                  runningIndex += 1
                  const idx = runningIndex
                  const active = idx === activeIndex
                  return (
                    <li
                      key={cmd.id}
                      id={`palette-item-${idx}`}
                      role="option"
                      aria-selected={active}
                      data-palette-item
                      className={active ? 'palette-item palette-item-active' : 'palette-item'}
                      onMouseMove={() => setActiveIndex(idx)}
                      onClick={() => void runAt(idx)}
                    >
                      <span className="palette-item-label">{cmd.label}</span>
                      {cmd.hint && <span className="palette-item-hint">{cmd.hint}</span>}
                    </li>
                  )
                })}
              </ul>
            </li>
          ))}
        </ul>
        <footer className="palette-footer">
          <span><Kbd>↑</Kbd><Kbd>↓</Kbd> navigate</span>
          <span><Kbd>↵</Kbd> run</span>
          <span><Kbd>Esc</Kbd> close</span>
        </footer>
      </div>
    </div>
  )
}

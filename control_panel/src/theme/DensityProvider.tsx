import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react'
import type { ReactNode } from 'react'

export type Density = 'comfortable' | 'compact'

type DensityContextValue = {
  density: Density
  setDensity: (next: Density) => void
  toggleDensity: () => void
}

const STORAGE_KEY = 'control-panel:density'

const DensityContext = createContext<DensityContextValue | null>(null)

function readInitialDensity(): Density {
  if (typeof window === 'undefined') return 'comfortable'
  const stored = window.localStorage.getItem(STORAGE_KEY)
  if (stored === 'comfortable' || stored === 'compact') return stored
  return 'comfortable'
}

function applyDensityAttribute(density: Density) {
  if (typeof document === 'undefined') return
  document.documentElement.setAttribute('data-density', density)
}

export function DensityProvider(props: { children: ReactNode; initialDensity?: Density }) {
  const [density, setDensityState] = useState<Density>(() => props.initialDensity ?? readInitialDensity())

  useEffect(() => {
    applyDensityAttribute(density)
    if (typeof window !== 'undefined') {
      window.localStorage.setItem(STORAGE_KEY, density)
    }
  }, [density])

  const setDensity = useCallback((next: Density) => setDensityState(next), [])
  const toggleDensity = useCallback(
    () => setDensityState(prev => (prev === 'comfortable' ? 'compact' : 'comfortable')),
    [],
  )

  const value = useMemo<DensityContextValue>(
    () => ({ density, setDensity, toggleDensity }),
    [density, setDensity, toggleDensity],
  )

  return <DensityContext.Provider value={value}>{props.children}</DensityContext.Provider>
}

export function useDensity(): DensityContextValue {
  const ctx = useContext(DensityContext)
  if (!ctx) {
    throw new Error('useDensity must be used within a DensityProvider')
  }
  return ctx
}

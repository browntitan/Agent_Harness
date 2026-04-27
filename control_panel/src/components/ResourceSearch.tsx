import { Search } from 'lucide-react'

export function ResourceSearch(props: {
  value: string
  onChange: (value: string) => void
  placeholder: string
}) {
  return (
    <label className="resource-search" aria-label={props.placeholder}>
      <Search size={14} strokeWidth={2} aria-hidden="true" />
      <input
        value={props.value}
        onChange={event => props.onChange(event.target.value)}
        placeholder={props.placeholder}
      />
      {props.value && (
        <button type="button" onClick={() => props.onChange('')} aria-label={`Clear ${props.placeholder.toLowerCase()}`}>
          ×
        </button>
      )}
    </label>
  )
}

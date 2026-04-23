# Control Panel Visual Assets

This folder holds the visual references for the control-panel handbook.

## Included In V1

- `control-panel-layout.svg`: a callout-style map of the shipped UI layout and section placement.
- `safe-first-tour.svg`: the recommended first-use review path for new users.
- `change-lifecycle.svg`: the main change flow from review through audit verification.

These visuals are diagram-style references built from the current shipped UI and labels. They are intentionally safe to commit and share because they avoid real tokens, file paths, and environment-specific data.

## Screenshot And Visual Rules

When you add live screenshots later, follow these rules:

- use seeded, non-sensitive sample data only
- mask tokens, secrets, user IDs, and file paths that should not be shared
- capture the shipped UI labels exactly as users see them
- prefer one clear screenshot per workflow over many near-duplicate crops
- keep screenshots paired with nearby text that explains why the view matters

## Recommended Future Screenshot Set

- login screen with masked token entry
- dashboard overview with masked runtime payload
- config validation preview
- agent editor and tool catalog
- prompt editor with overlay save and reset actions
- collections view with document viewer
- skills view with preview and status toggle
- operations view with reload and audit history

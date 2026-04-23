---
name: Return File Handoff
agent_scope: data_analyst
tool_tags: workspace_write, return_file, workspace_list
task_tags: files, artifact, handoff
version: 2
enabled: true
description: Publish derived analyst outputs explicitly so the user receives a usable artifact instead of only a text description.
keywords: return file, artifact, handoff
when_to_apply: Use when analysis generates a workbook, CSV, or other deliverable file.
avoid_when: Avoid ending the turn without returning the file when the deliverable matters.
examples: cleaned workbook, labeled CSV, chart image
---
# Return File Handoff

## Rule

When the output is a file, register it with `return_file` and mention it clearly in the final response.

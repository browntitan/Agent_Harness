export interface AdminField {
  env_name: string
  label: string
  group: string
  description: string
  kind: string
  choices: string[]
  secret: boolean
  readonly: boolean
  optional?: boolean
  reload_scope: string
  ui_control?: string
  min_value?: number | null
  max_value?: number | null
  step?: number | null
  value: string
  is_configured: boolean
}

export interface AdminOverview {
  status: string
  gateway_model_id: string
  providers: Record<string, string>
  models: Record<string, string>
  counts: Record<string, number>
  collections: Array<Record<string, unknown>>
  agents: Array<Record<string, unknown>>
  jobs: Array<Record<string, unknown>>
  last_reload: Record<string, unknown>
  audit_events: Array<Record<string, unknown>>
}

export interface ConfigValidationResult {
  valid: boolean
  errors?: Record<string, string>
  normalized_changes?: Record<string, string | null>
  preview_diff?: Record<string, { before: string; after: string }>
  reload_scope?: string
  applied?: boolean
  reload?: Record<string, unknown>
}

export interface CapabilitySectionStatus {
  supported: boolean
  required_routes: string[]
  missing_routes: string[]
  reason: string
}

export interface ControlPanelCapabilities {
  schema_version: string
  contract_version: string
  compatible: boolean
  generated_at: string
  sections: Record<string, CapabilitySectionStatus>
}

export type ServiceResetEngine = 'docker' | 'podman'

export interface ServiceResetResult {
  status: string
  run_id: string
  engine: ServiceResetEngine
  started_at: string
  actor?: string
  repo_root?: string
  log_path: string
  status_path?: string
  exit_code?: string
  pid?: number
  commands: string[]
}

export interface AccessPrincipal {
  principal_id: string
  tenant_id: string
  principal_type: string
  provider: string
  external_id: string
  email_normalized: string
  display_name: string
  metadata_json: Record<string, unknown>
  active: boolean
  created_at: string
  updated_at: string
}

export interface AccessMembership {
  membership_id: string
  tenant_id: string
  parent_principal_id: string
  child_principal_id: string
  created_at: string
}

export interface AccessRole {
  role_id: string
  tenant_id: string
  name: string
  description: string
  created_at: string
  updated_at: string
}

export interface AccessRoleBinding {
  binding_id: string
  tenant_id: string
  role_id: string
  principal_id: string
  created_at: string
  disabled_at: string
  disabled: boolean
}

export interface AccessRolePermission {
  permission_id: string
  tenant_id: string
  role_id: string
  resource_type: string
  action: string
  resource_selector: string
  created_at: string
}

export interface EffectiveAccessPayload {
  email: string
  access: Record<string, unknown>
}

export interface McpToolCatalogRecord {
  tool_id: string
  connection_id: string
  registry_name: string
  raw_tool_name: string
  description: string
  input_schema: Record<string, unknown>
  read_only: boolean
  destructive: boolean
  background_safe: boolean
  should_defer: boolean
  enabled: boolean
  status: string
  search_hint: string
  defer_priority: number
  metadata_json?: Record<string, unknown>
}

export interface McpConnectionRecord {
  connection_id: string
  display_name: string
  connection_slug: string
  server_url: string
  auth_type: string
  status: string
  allowed_agents: string[]
  visibility: string
  health: Record<string, unknown>
  secret_configured: boolean
  owner_user_id: string
  tenant_id: string
  metadata_json?: Record<string, unknown>
  last_tested_at: string
  last_refreshed_at: string
  tools?: McpToolCatalogRecord[]
}

export interface ArchitectureNode {
  id: string
  label: string
  kind: string
  layer: string
  description: string
  status: string
  mode?: string
  role_kind?: string
  entry_path?: string
  prompt_file?: string
  overlay_active?: boolean
  allowed_tools?: string[]
  allowed_worker_agents?: string[]
  preload_skill_packs?: string[]
  memory_scopes?: string[]
  badges?: string[]
}

export interface ArchitectureEdge {
  id: string
  source: string
  target: string
  kind: string
  label?: string
  emphasis?: string
}

export interface CanonicalRoutingPath {
  id: string
  label: string
  route: string
  summary: string
  when?: string
  target_agent?: string
  badges?: string[]
  node_ids?: string[]
  edge_ids?: string[]
}

export interface LangGraphExport {
  status?: string
  generated_at?: string
  agent_name?: string
  mermaid?: string
  nodes?: Record<string, unknown>[]
  edges?: Record<string, unknown>[]
  warnings?: string[]
}

export interface ArchitectureSnapshot {
  generated_at: string
  system: Record<string, unknown>
  router: Record<string, unknown>
  nodes: ArchitectureNode[]
  edges: ArchitectureEdge[]
  canonical_paths: CanonicalRoutingPath[]
  langgraph?: LangGraphExport
}

export interface ArchitectureFlow {
  session_id: string
  conversation_id?: string
  route?: string
  router_method?: string
  start_agent?: string
  suggested_agent?: string
  reasons?: string[]
  worker_agents?: string[]
  degraded?: boolean
  degraded_events?: string[]
  updated_at?: string
  started_at?: string
}

export interface ArchitectureActivity {
  route_counts: Record<string, number>
  router_method_counts: Record<string, number>
  start_agent_counts: Record<string, number>
  delegation_counts: Record<string, number>
  outcome_counts: Record<string, number>
  negative_rate_by_route: Record<string, number>
  negative_rate_by_router_method: Record<string, number>
  recent_mispicks: Array<Record<string, unknown>>
  review_backlog: Record<string, unknown>
  last_retrain_report: Record<string, unknown>
  recent_flows: ArchitectureFlow[]
  updated_at: string
}

export interface CollectionHealthRecord {
  doc_id: string
  title: string
  source_type: string
  source_path: string
  source_display_path?: string
  collection_id: string
  content_hash: string
  ingested_at: string
  num_chunks: number
  file_type: string
  doc_structure_type: string
  active: boolean
  version_ordinal?: number
  superseded_at?: string
  parser_chain?: string[]
  extraction_status?: string
  extraction_error?: string
  metadata_confidence?: number
  lifecycle_phase?: string
  doc_type?: string
  program_entities?: string[]
  signal_summary?: Record<string, unknown>
}

export interface CollectionHealthGroup {
  source_identity: string
  title: string
  source_type: string
  collection_id: string
  configured_source_path: string
  active_doc_id: string
  active_content_hash: string
  active_ingested_at: string
  active_source_path: string
  current_file_hash: string
  source_exists: boolean
  content_drift: boolean
  duplicate_doc_ids: string[]
  stale_version_doc_ids?: string[]
  extraction_failure_doc_ids?: string[]
  low_confidence_doc_ids?: string[]
  missing_source_doc_ids?: string[]
  parser_warnings?: string[]
  status: string
  records: CollectionHealthRecord[]
}

export interface CollectionHealthReport {
  status: string
  reason: string
  tenant_id: string
  collection_id: string
  maintenance_policy?: string
  configured_source_count: number
  indexed_doc_count: number
  active_doc_count: number
  missing_sources: string[]
  duplicate_group_count: number
  content_drift_count: number
  stale_version_count?: number
  extraction_failure_count?: number
  low_confidence_metadata_count?: number
  missing_source_doc_count?: number
  parser_warning_count?: number
  parser_counts?: Record<string, number>
  doc_type_counts?: Record<string, number>
  lifecycle_counts?: Record<string, number>
  metadata_confidence_distribution?: Record<string, number>
  duplicate_groups: CollectionHealthGroup[]
  drifted_groups: CollectionHealthGroup[]
  source_groups: CollectionHealthGroup[]
  sync_error: string
  suggested_fix: string
}

export interface CollectionStatusSummary {
  ready: boolean
  reason: string
  collection_id: string
  missing_sources: string[]
  indexed_doc_count: number
  active_doc_count: number
  duplicate_group_count: number
  content_drift_count: number
  suggested_fix: string
}

export interface CollectionStorageProfile {
  vector_store_backend: string
  tables: string[]
  embeddings_provider: string
  embedding_model: string
  graph_embedding_model?: string
  configured_embedding_dim: number
  actual_embedding_dims: Record<string, number>
  mismatch_warnings: string[]
}

export interface CollectionSummary {
  collection_id: string
  created_at: string
  updated_at: string
  maintenance_policy: string
  document_count: number
  source_type_counts: Record<string, number>
  latest_ingested_at: string
  graph_count: number
  graph_ids: string[]
  storage_profile: CollectionStorageProfile
  status: CollectionStatusSummary
}

export interface UploadedFileSummary {
  doc_id: string
  title: string
  source_type: string
  source_path: string
  source_display_path?: string
  collection_id: string
  num_chunks: number
  ingested_at: string
  file_type: string
  doc_structure_type: string
  source_metadata?: Record<string, unknown>
  metadata_summary?: Record<string, unknown>
}

export interface CollectionOperationFile {
  display_path: string
  filename: string
  source_type: string
  source_path?: string
  outcome: string
  error?: string
  doc_ids: string[]
  extraction_status?: string
  metadata_confidence?: number
  parser_provenance?: Record<string, unknown>
  metadata_summary?: Record<string, unknown>
}

export interface CollectionOperationSummary {
  resolved_count: number
  ingested_count: number
  already_indexed_count?: number
  skipped_count: number
  failed_count: number
  missing_count: number
}

export interface CollectionOperationResult {
  collection_id: string
  status: string
  summary: CollectionOperationSummary
  resolved_count: number
  ingested_count: number
  already_indexed_count?: number
  skipped_count: number
  failed_count: number
  doc_ids: string[]
  missing_paths: string[]
  errors: string[]
  files: CollectionOperationFile[]
  filenames: string[]
  display_paths: string[]
  workspace_copies?: string[]
  collection_status?: CollectionStatusSummary
  metadata_summary?: Record<string, unknown>
}

export interface SourceScanSummary {
  supported_count: number
  skipped_count: number
  missing_count: number
  blocked_count: number
  duplicate_display_path_count: number
  duplicate_filename_count: number
  total_size_bytes: number
  estimated_chunks: number
}

export interface SourceScanFile {
  display_path: string
  filename: string
  source_path: string
  source_type: string
  supported: boolean
  outcome: string
  error?: string
  duplicate_display_path?: boolean
  size_bytes?: number
  fingerprint?: Record<string, unknown>
}

export interface SourceScanPayload {
  object: string
  status: string
  source_kind: string
  collection_id: string
  metadata_profile: string
  requested_paths: string[]
  allowed_roots: string[]
  roots: Array<Record<string, unknown>>
  summary: SourceScanSummary
  files: SourceScanFile[]
  supported_files: SourceScanFile[]
  skipped_files: SourceScanFile[]
  missing_paths: string[]
  blocked_paths: string[]
  duplicate_display_paths: string[]
  duplicate_filenames: string[]
  warnings: string[]
}

export interface RegisteredSource {
  source_id: string
  tenant_id?: string
  collection_id: string
  display_name: string
  source_kind: string
  paths: string[]
  include_globs?: string[]
  exclude_globs?: string[]
  last_scan?: SourceScanPayload
  last_refresh?: Record<string, unknown>
  created_at?: string
  updated_at?: string
}

export interface SourceRefreshRun {
  run_id: string
  source_id: string
  operation: string
  status: string
  detail?: string
  started_at?: string
  completed_at?: string
  updated_at?: string
  result?: Record<string, unknown>
}

export interface GraphAssistantPayload {
  friendly?: Record<string, unknown>
  validation?: Record<string, unknown>
  result?: Record<string, unknown>
  graph_id?: string
  query?: string
  display_name?: string
  config_overrides?: Record<string, unknown>
  prompt_overrides?: Record<string, unknown>
  source_doc_ids?: string[]
  source_count?: number
  guidance?: string
}

export interface GraphIndexRecord {
  graph_id: string
  tenant_id: string
  collection_id: string
  display_name: string
  owner_admin_user_id?: string
  visibility?: string
  backend: string
  status: string
  root_path?: string
  artifact_path?: string
  domain_summary?: string
  entity_samples?: string[]
  relationship_samples?: string[]
  source_doc_ids?: string[]
  graph_skill_ids?: string[]
  query_ready?: boolean
  query_backend?: string
  artifact_tables?: string[]
  artifact_mtime?: string
  graph_context_summary?: Record<string, unknown>
  config_json?: Record<string, unknown>
  prompt_overrides_json?: Record<string, unknown>
  health?: Record<string, unknown>
  freshness_score?: number
  last_indexed_at?: string
  updated_at?: string
}

export interface GraphIndexSourceRecord {
  graph_source_id: string
  graph_id: string
  source_doc_id: string
  source_path: string
  source_title: string
  source_type: string
  created_at?: string
}

export interface GraphIndexRunRecord {
  run_id: string
  graph_id: string
  operation: string
  status: string
  detail: string
  metadata?: Record<string, unknown>
  started_at?: string
  completed_at?: string
}

export interface GraphLogRecord {
  path: string
  name: string
  size_bytes: number
  modified_at: string
  preview: string
}

export interface GraphProgressStage {
  id: string
  label: string
  state: string
  workflow?: string
}

export interface GraphProgressPayload {
  graph_id: string
  status: string
  active: boolean
  active_run?: GraphIndexRunRecord | null
  latest_run?: GraphIndexRunRecord | null
  workflow?: string
  task_progress?: Record<string, unknown>
  stages: GraphProgressStage[]
  percent: number
  updated_at?: string
  logs?: GraphLogRecord[]
  log_tail?: string
  cursor?: string
}

export interface GraphDetailPayload {
  graph: GraphIndexRecord
  sources: GraphIndexSourceRecord[]
  runs: GraphIndexRunRecord[]
  logs?: GraphLogRecord[]
  skills?: Array<Record<string, unknown>>
  warnings?: string[]
}

export interface GraphResearchTunePayload {
  run_id: string
  graph_id: string
  status: string
  detail?: string
  artifact_dir?: string
  manifest_path?: string
  scratchpad_path?: string
  scratchpad_preview?: string
  manifest?: Record<string, unknown>
  coverage?: Record<string, unknown>
  warnings?: string[]
  corpus_profile?: Record<string, unknown>
  doc_digests?: Array<Record<string, unknown>>
  prompt_drafts?: Record<string, Record<string, unknown>>
  prompt_diffs?: Record<string, Record<string, unknown>>
}

export interface CollectionSkillDraftRecord {
  draft_type: string
  label: string
  skill_id: string
  name: string
  agent_scope: string
  collection_id: string
  graph_id?: string
  body_markdown: string
  markdown?: string
  selected?: boolean
  description?: string
  tool_tags?: string[]
  task_tags?: string[]
  controller_hints?: Record<string, unknown>
}

export interface CollectionSkillDraftPayload {
  object: string
  collection_id: string
  graph_id?: string
  drafts: CollectionSkillDraftRecord[]
  mutated?: boolean
}

export interface SkillBuildDraftRecord {
  body_markdown: string
  name: string
  agent_scope: string
  description: string
  tool_tags: string[]
  task_tags: string[]
  when_to_apply: string
  workflow: string
  examples: string
  warnings?: string[]
}

export interface SkillBuildDraftPayload {
  object: string
  draft: SkillBuildDraftRecord
}

export interface CollectionSkillDraftApplyPayload {
  object: string
  collection_id: string
  graph_id?: string
  applied_skill_ids: string[]
  graph_bound_skill_ids: string[]
  skills: Array<Record<string, unknown>>
}

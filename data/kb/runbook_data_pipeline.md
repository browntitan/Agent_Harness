# Runbook — Data Pipeline Operations

**Owner:** Data Engineering
**Last reviewed:** 2025-09-01

## 1. Overview

This runbook describes operational procedures for the daily batch pipeline.

## 2. SLA

- Daily batch completes by 06:00 UTC

## 3. Common failures

### 3.1 Upstream source delay

Symptoms:

- missing partitions
- jobs waiting on input

Mitigation:

- confirm upstream status
- if delay > 2h, page upstream owner

### 3.2 Schema mismatch

Symptoms:

- parquet read errors
- schema evolution errors

Mitigation:

- rollback schema changes
- run backfill with pinned schema

## 4. Backfill procedure

1) Identify missing range
2) Confirm compute capacity
3) Run backfill job
4) Validate metrics
5) Communicate completion


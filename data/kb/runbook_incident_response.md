# Runbook — Incident Response

**Owner:** SRE
**Last reviewed:** 2025-12-15

## 1. Overview

This runbook defines severity levels, roles, and procedures for incident response.

## 2. Severity levels

### 2.1 SEV-1

A SEV-1 is a **user-visible outage** impacting a core customer flow.

Examples:

- login is broken for most users
- checkout fails across regions
- data loss is suspected

Required actions:

- Page the **Incident Commander (IC)** immediately
- Page the **Communications Lead**
- Establish an incident channel
- Start an incident timeline

**Update cadence:** post an update every **15 minutes** until mitigated.

### 2.2 SEV-2

A SEV-2 is a **degraded** experience or partial outage.

Examples:

- elevated error rates for a subset of users
- partial region outage with failover

Update cadence:

- post an update every **30 minutes**

### 2.3 SEV-3

A SEV-3 is a minor issue or internal degradation with limited user impact.

Update cadence:

- post updates as needed

## 3. Roles

- Incident Commander (IC)
- Ops Lead
- Comms Lead
- Subject Matter Experts (SMEs)

## 4. Initial checklist

1) Declare severity
2) Assign IC
3) Create incident channel
4) Start timeline
5) Mitigation plan

## 5. Post-incident

- Write postmortem within 5 business days
- Identify action items
- Track action items to completion


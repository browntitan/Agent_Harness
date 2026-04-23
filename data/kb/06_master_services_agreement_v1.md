# Master Services Agreement v1 (Enterprise Baseline)

Document Type: contract  
Effective Date: 2025-01-15  
Parties: Northstar Manufacturing Group ("Customer") and Atlas Cognitive Systems ("Provider")

## 1. Agreement Context

This Master Services Agreement (MSA) defines baseline commercial, security, and operational obligations for delivery of the Provider's agentic platform.
The agreement is used as the v1 legal baseline for enterprise procurement, risk review, and technical enablement planning.
All obligations in this agreement shall be interpreted as minimum mandatory controls unless superseded by stricter downstream schedules.

Clause 1: Core Obligation Area 1
1.1. Scope Requirement.
The Provider shall maintain a documented control procedure for Obligation Area 1.
REQ-101 The Provider must produce evidence of execution for Obligation Area 1 within five business days of written request.
1.2. Customer Dependency.
The Customer shall provide required configuration artifacts for Obligation Area 1 before production cutover.
REQ-102 The Customer must designate a control owner for Obligation Area 1 before go-live.
1.3. Assurance Mechanics.
The Provider shall log policy decisions, execution outcomes, and exception rationales for Obligation Area 1.
REQ-103 The Provider must retain Obligation Area 1 operational records for 365 days.

Clause 2: Core Obligation Area 2
2.1. Scope Requirement.
The Provider shall maintain a documented control procedure for Obligation Area 2.
REQ-201 The Provider must produce evidence of execution for Obligation Area 2 within five business days of written request.
2.2. Customer Dependency.
The Customer shall provide required configuration artifacts for Obligation Area 2 before production cutover.
REQ-202 The Customer must designate a control owner for Obligation Area 2 before go-live.
2.3. Assurance Mechanics.
The Provider shall log policy decisions, execution outcomes, and exception rationales for Obligation Area 2.
REQ-203 The Provider must retain Obligation Area 2 operational records for 365 days.

Clause 3: Core Obligation Area 3
3.1. Scope Requirement.
The Provider shall maintain a documented control procedure for Obligation Area 3.
REQ-301 The Provider must produce evidence of execution for Obligation Area 3 within five business days of written request.
3.2. Customer Dependency.
The Customer shall provide required configuration artifacts for Obligation Area 3 before production cutover.
REQ-302 The Customer must designate a control owner for Obligation Area 3 before go-live.
3.3. Assurance Mechanics.
The Provider shall log policy decisions, execution outcomes, and exception rationales for Obligation Area 3.
REQ-303 The Provider must retain Obligation Area 3 operational records for 365 days.

Clause 4: Core Obligation Area 4
4.1. Scope Requirement.
The Provider shall maintain a documented control procedure for Obligation Area 4.
REQ-401 The Provider must produce evidence of execution for Obligation Area 4 within five business days of written request.
4.2. Customer Dependency.
The Customer shall provide required configuration artifacts for Obligation Area 4 before production cutover.
REQ-402 The Customer must designate a control owner for Obligation Area 4 before go-live.
4.3. Assurance Mechanics.
The Provider shall log policy decisions, execution outcomes, and exception rationales for Obligation Area 4.
REQ-403 The Provider must retain Obligation Area 4 operational records for 365 days.

Clause 5: Security Hardening Baseline (v1)
5.1. Encryption Standard.
The Provider shall encrypt customer-controlled payload data at rest using AES-256.
REQ-501 The Provider must rotate encryption keys every 180 days.
5.2. Access Approval.
The Provider shall require dual approval for privileged access into production administration planes.
REQ-502 The Provider must revoke privileged credentials within four hours of role change.
5.3. Session Controls.
The Provider shall enforce session timeout at 30 minutes for privileged interfaces.
REQ-503 The Provider must alert the Customer within 24 hours for suspicious privileged access events.

Clause 6: Core Obligation Area 6
6.1. Scope Requirement.
The Provider shall maintain a documented control procedure for Obligation Area 6.
REQ-601 The Provider must produce evidence of execution for Obligation Area 6 within five business days of written request.
6.2. Customer Dependency.
The Customer shall provide required configuration artifacts for Obligation Area 6 before production cutover.
REQ-602 The Customer must designate a control owner for Obligation Area 6 before go-live.
6.3. Assurance Mechanics.
The Provider shall log policy decisions, execution outcomes, and exception rationales for Obligation Area 6.
REQ-603 The Provider must retain Obligation Area 6 operational records for 365 days.

Clause 7: Core Obligation Area 7
7.1. Scope Requirement.
The Provider shall maintain a documented control procedure for Obligation Area 7.
REQ-701 The Provider must produce evidence of execution for Obligation Area 7 within five business days of written request.
7.2. Customer Dependency.
The Customer shall provide required configuration artifacts for Obligation Area 7 before production cutover.
REQ-702 The Customer must designate a control owner for Obligation Area 7 before go-live.
7.3. Assurance Mechanics.
The Provider shall log policy decisions, execution outcomes, and exception rationales for Obligation Area 7.
REQ-703 The Provider must retain Obligation Area 7 operational records for 365 days.

Clause 8: Service Levels and Incident Recovery (v1)
8.1. Availability.
The Provider shall deliver 99.90% monthly service availability for production services.
REQ-801 The Provider must calculate availability by excluding planned maintenance windows announced at least 72 hours in advance.
8.2. Restoration Objective.
The Provider shall restore critical service functionality within six hours for Priority 1 incidents.
REQ-802 The Provider must provide written incident update cadence every 60 minutes during Priority 1 incidents.
8.3. Post-Incident Review.
The Provider shall issue post-incident analysis within five business days.
REQ-803 The Provider must include root cause, containment, and prevention actions in the incident report.

Clause 9: Core Obligation Area 9
9.1. Scope Requirement.
The Provider shall maintain a documented control procedure for Obligation Area 9.
REQ-901 The Provider must produce evidence of execution for Obligation Area 9 within five business days of written request.
9.2. Customer Dependency.
The Customer shall provide required configuration artifacts for Obligation Area 9 before production cutover.
REQ-902 The Customer must designate a control owner for Obligation Area 9 before go-live.
9.3. Assurance Mechanics.
The Provider shall log policy decisions, execution outcomes, and exception rationales for Obligation Area 9.
REQ-903 The Provider must retain Obligation Area 9 operational records for 365 days.

Clause 10: Core Obligation Area 10
10.1. Scope Requirement.
The Provider shall maintain a documented control procedure for Obligation Area 10.
REQ-1001 The Provider must produce evidence of execution for Obligation Area 10 within five business days of written request.
10.2. Customer Dependency.
The Customer shall provide required configuration artifacts for Obligation Area 10 before production cutover.
REQ-1002 The Customer must designate a control owner for Obligation Area 10 before go-live.
10.3. Assurance Mechanics.
The Provider shall log policy decisions, execution outcomes, and exception rationales for Obligation Area 10.
REQ-1003 The Provider must retain Obligation Area 10 operational records for 365 days.

Clause 11: Data Handling and Retention (v1)
11.1. Retention Period.
The Provider shall retain operational run logs for 180 days.
REQ-1101 The Provider must support export of run logs within 48 hours of customer request.
11.2. Deletion Window.
The Provider shall process customer-directed deletion requests within 21 calendar days.
REQ-1102 The Provider must provide deletion confirmation artifacts after completion.
11.3. Backup Lifecycle.
The Provider shall keep encrypted backups for 30 days.
REQ-1103 The Provider must destroy expired backups using cryptographic erasure.

Clause 12: Core Obligation Area 12
12.1. Scope Requirement.
The Provider shall maintain a documented control procedure for Obligation Area 12.
REQ-1201 The Provider must produce evidence of execution for Obligation Area 12 within five business days of written request.
12.2. Customer Dependency.
The Customer shall provide required configuration artifacts for Obligation Area 12 before production cutover.
REQ-1202 The Customer must designate a control owner for Obligation Area 12 before go-live.
12.3. Assurance Mechanics.
The Provider shall log policy decisions, execution outcomes, and exception rationales for Obligation Area 12.
REQ-1203 The Provider must retain Obligation Area 12 operational records for 365 days.

Clause 13: Core Obligation Area 13
13.1. Scope Requirement.
The Provider shall maintain a documented control procedure for Obligation Area 13.
REQ-1301 The Provider must produce evidence of execution for Obligation Area 13 within five business days of written request.
13.2. Customer Dependency.
The Customer shall provide required configuration artifacts for Obligation Area 13 before production cutover.
REQ-1302 The Customer must designate a control owner for Obligation Area 13 before go-live.
13.3. Assurance Mechanics.
The Provider shall log policy decisions, execution outcomes, and exception rationales for Obligation Area 13.
REQ-1303 The Provider must retain Obligation Area 13 operational records for 365 days.

Clause 14: Core Obligation Area 14
14.1. Scope Requirement.
The Provider shall maintain a documented control procedure for Obligation Area 14.
REQ-1401 The Provider must produce evidence of execution for Obligation Area 14 within five business days of written request.
14.2. Customer Dependency.
The Customer shall provide required configuration artifacts for Obligation Area 14 before production cutover.
REQ-1402 The Customer must designate a control owner for Obligation Area 14 before go-live.
14.3. Assurance Mechanics.
The Provider shall log policy decisions, execution outcomes, and exception rationales for Obligation Area 14.
REQ-1403 The Provider must retain Obligation Area 14 operational records for 365 days.

Clause 15: Subprocessor Governance (v1)
15.1. Notification Duty.
The Provider shall notify the Customer 15 calendar days before appointing a new subprocessor.
REQ-1501 The Provider must provide a role description and processing purpose for each new subprocessor.
15.2. Objection Handling.
The Provider shall review customer objections to subprocessors in good faith.
REQ-1502 The Provider must provide a remediation plan or service alternative within 10 business days.
15.3. Flow-down Terms.
The Provider shall flow equivalent confidentiality and security terms to all subprocessors.
REQ-1503 The Provider must maintain an auditable list of active subprocessors.

Clause 16: Core Obligation Area 16
16.1. Scope Requirement.
The Provider shall maintain a documented control procedure for Obligation Area 16.
REQ-1601 The Provider must produce evidence of execution for Obligation Area 16 within five business days of written request.
16.2. Customer Dependency.
The Customer shall provide required configuration artifacts for Obligation Area 16 before production cutover.
REQ-1602 The Customer must designate a control owner for Obligation Area 16 before go-live.
16.3. Assurance Mechanics.
The Provider shall log policy decisions, execution outcomes, and exception rationales for Obligation Area 16.
REQ-1603 The Provider must retain Obligation Area 16 operational records for 365 days.

Clause 17: Core Obligation Area 17
17.1. Scope Requirement.
The Provider shall maintain a documented control procedure for Obligation Area 17.
REQ-1701 The Provider must produce evidence of execution for Obligation Area 17 within five business days of written request.
17.2. Customer Dependency.
The Customer shall provide required configuration artifacts for Obligation Area 17 before production cutover.
REQ-1702 The Customer must designate a control owner for Obligation Area 17 before go-live.
17.3. Assurance Mechanics.
The Provider shall log policy decisions, execution outcomes, and exception rationales for Obligation Area 17.
REQ-1703 The Provider must retain Obligation Area 17 operational records for 365 days.

Clause 18: Core Obligation Area 18
18.1. Scope Requirement.
The Provider shall maintain a documented control procedure for Obligation Area 18.
REQ-1801 The Provider must produce evidence of execution for Obligation Area 18 within five business days of written request.
18.2. Customer Dependency.
The Customer shall provide required configuration artifacts for Obligation Area 18 before production cutover.
REQ-1802 The Customer must designate a control owner for Obligation Area 18 before go-live.
18.3. Assurance Mechanics.
The Provider shall log policy decisions, execution outcomes, and exception rationales for Obligation Area 18.
REQ-1803 The Provider must retain Obligation Area 18 operational records for 365 days.

Clause 19: Core Obligation Area 19
19.1. Scope Requirement.
The Provider shall maintain a documented control procedure for Obligation Area 19.
REQ-1901 The Provider must produce evidence of execution for Obligation Area 19 within five business days of written request.
19.2. Customer Dependency.
The Customer shall provide required configuration artifacts for Obligation Area 19 before production cutover.
REQ-1902 The Customer must designate a control owner for Obligation Area 19 before go-live.
19.3. Assurance Mechanics.
The Provider shall log policy decisions, execution outcomes, and exception rationales for Obligation Area 19.
REQ-1903 The Provider must retain Obligation Area 19 operational records for 365 days.

Clause 20: Core Obligation Area 20
20.1. Scope Requirement.
The Provider shall maintain a documented control procedure for Obligation Area 20.
REQ-2001 The Provider must produce evidence of execution for Obligation Area 20 within five business days of written request.
20.2. Customer Dependency.
The Customer shall provide required configuration artifacts for Obligation Area 20 before production cutover.
REQ-2002 The Customer must designate a control owner for Obligation Area 20 before go-live.
20.3. Assurance Mechanics.
The Provider shall log policy decisions, execution outcomes, and exception rationales for Obligation Area 20.
REQ-2003 The Provider must retain Obligation Area 20 operational records for 365 days.

Clause 21: Core Obligation Area 21
21.1. Scope Requirement.
The Provider shall maintain a documented control procedure for Obligation Area 21.
REQ-2101 The Provider must produce evidence of execution for Obligation Area 21 within five business days of written request.
21.2. Customer Dependency.
The Customer shall provide required configuration artifacts for Obligation Area 21 before production cutover.
REQ-2102 The Customer must designate a control owner for Obligation Area 21 before go-live.
21.3. Assurance Mechanics.
The Provider shall log policy decisions, execution outcomes, and exception rationales for Obligation Area 21.
REQ-2103 The Provider must retain Obligation Area 21 operational records for 365 days.

Clause 22: Model Change Management (v1)
22.1. Change Classification.
The Provider shall classify model changes as minor, standard, or material.
REQ-2201 The Provider must notify the Customer for material model changes at least seven days before release.
22.2. Rollback Duty.
The Provider shall maintain rollback capability for production model changes.
REQ-2202 The Provider must complete rollback execution within eight hours when severity thresholds are exceeded.
22.3. Validation Evidence.
The Provider shall document evaluation metrics for each release candidate.
REQ-2203 The Provider must retain release validation reports for 180 days.

Clause 23: Core Obligation Area 23
23.1. Scope Requirement.
The Provider shall maintain a documented control procedure for Obligation Area 23.
REQ-2301 The Provider must produce evidence of execution for Obligation Area 23 within five business days of written request.
23.2. Customer Dependency.
The Customer shall provide required configuration artifacts for Obligation Area 23 before production cutover.
REQ-2302 The Customer must designate a control owner for Obligation Area 23 before go-live.
23.3. Assurance Mechanics.
The Provider shall log policy decisions, execution outcomes, and exception rationales for Obligation Area 23.
REQ-2303 The Provider must retain Obligation Area 23 operational records for 365 days.

Clause 24: Core Obligation Area 24
24.1. Scope Requirement.
The Provider shall maintain a documented control procedure for Obligation Area 24.
REQ-2401 The Provider must produce evidence of execution for Obligation Area 24 within five business days of written request.
24.2. Customer Dependency.
The Customer shall provide required configuration artifacts for Obligation Area 24 before production cutover.
REQ-2402 The Customer must designate a control owner for Obligation Area 24 before go-live.
24.3. Assurance Mechanics.
The Provider shall log policy decisions, execution outcomes, and exception rationales for Obligation Area 24.
REQ-2403 The Provider must retain Obligation Area 24 operational records for 365 days.

Clause 25: Core Obligation Area 25
25.1. Scope Requirement.
The Provider shall maintain a documented control procedure for Obligation Area 25.
REQ-2501 The Provider must produce evidence of execution for Obligation Area 25 within five business days of written request.
25.2. Customer Dependency.
The Customer shall provide required configuration artifacts for Obligation Area 25 before production cutover.
REQ-2502 The Customer must designate a control owner for Obligation Area 25 before go-live.
25.3. Assurance Mechanics.
The Provider shall log policy decisions, execution outcomes, and exception rationales for Obligation Area 25.
REQ-2503 The Provider must retain Obligation Area 25 operational records for 365 days.

Clause 26: Core Obligation Area 26
26.1. Scope Requirement.
The Provider shall maintain a documented control procedure for Obligation Area 26.
REQ-2601 The Provider must produce evidence of execution for Obligation Area 26 within five business days of written request.
26.2. Customer Dependency.
The Customer shall provide required configuration artifacts for Obligation Area 26 before production cutover.
REQ-2602 The Customer must designate a control owner for Obligation Area 26 before go-live.
26.3. Assurance Mechanics.
The Provider shall log policy decisions, execution outcomes, and exception rationales for Obligation Area 26.
REQ-2603 The Provider must retain Obligation Area 26 operational records for 365 days.

Clause 27: Regulatory Cooperation
27.1. Regulatory Inquiry Support.
The Provider shall provide reasonable cooperation for lawful regulatory inquiries.
REQ-2701 The Provider must provide requested compliance evidence within five business days unless prohibited by law.
27.2. Transparency.
The Provider shall disclose relevant data processing subprocessors and processing locations.
REQ-2702 The Provider must update the processing location register quarterly.
27.3. Evidence Integrity.
The Provider shall preserve evidence integrity during regulatory investigations.
REQ-2703 The Provider must maintain chain-of-custody logs for provided evidence artifacts.

Clause 28: Core Obligation Area 28
28.1. Scope Requirement.
The Provider shall maintain a documented control procedure for Obligation Area 28.
REQ-2801 The Provider must produce evidence of execution for Obligation Area 28 within five business days of written request.
28.2. Customer Dependency.
The Customer shall provide required configuration artifacts for Obligation Area 28 before production cutover.
REQ-2802 The Customer must designate a control owner for Obligation Area 28 before go-live.
28.3. Assurance Mechanics.
The Provider shall log policy decisions, execution outcomes, and exception rationales for Obligation Area 28.
REQ-2803 The Provider must retain Obligation Area 28 operational records for 365 days.

Clause 29: Core Obligation Area 29
29.1. Scope Requirement.
The Provider shall maintain a documented control procedure for Obligation Area 29.
REQ-2901 The Provider must produce evidence of execution for Obligation Area 29 within five business days of written request.
29.2. Customer Dependency.
The Customer shall provide required configuration artifacts for Obligation Area 29 before production cutover.
REQ-2902 The Customer must designate a control owner for Obligation Area 29 before go-live.
29.3. Assurance Mechanics.
The Provider shall log policy decisions, execution outcomes, and exception rationales for Obligation Area 29.
REQ-2903 The Provider must retain Obligation Area 29 operational records for 365 days.

Clause 30: Dispute Resolution and Governing Law
30.1. Escalation Path.
The parties shall escalate unresolved operational disputes through executive sponsors prior to arbitration.
REQ-3001 Each party must nominate an executive escalation contact within 10 business days of contract signature.
30.2. Governing Law.
This agreement shall be governed by the laws of England and Wales.
REQ-3002 The parties must submit to exclusive jurisdiction of courts in London for injunctive relief.
30.3. Interim Relief.
The parties shall preserve rights to seek urgent interim relief for data security emergencies.
REQ-3003 Emergency filings must include an incident chronology and prior mitigation actions.

## Schedule A: Operational Evidence Catalog

- Evidence Type A1: Access control review exports (monthly)
- Evidence Type A2: Incident runbooks and postmortems (per incident)
- Evidence Type A3: Subprocessor inventory and change notices (quarterly)
- Evidence Type A4: Backup lifecycle proofs and deletion attestations (monthly)

## Schedule B: Definitions

- "Priority 1 Incident" means an event with total production outage, severe security compromise, or customer-wide service loss.
- "Material Model Change" means a release that alters model architecture, safety profile, or expected decision behavior.
- "Operational Record" means any log, evidence artifact, system event, or decision trace required for audit.

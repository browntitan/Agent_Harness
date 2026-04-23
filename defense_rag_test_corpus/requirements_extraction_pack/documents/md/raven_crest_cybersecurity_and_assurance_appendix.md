# UNCLASSIFIED // SYNTHETIC TEST DATA
# Raven Crest Cybersecurity and Assurance Appendix

Program: Raven Crest Expeditionary Relay Modernization  
Revision: Rev A  
Date: 18 Mar 2031


## 1. Purpose
This appendix captures broader mandatory cybersecurity and assurance language intended to exercise `mandatory` extraction mode. It intentionally mixes `must`, `must not`, `required to`, prohibited forms, and a small number of `shall` statements. Narrative paragraphs, examples, planning notes, and risk discussions are included nearby so the extractor is tested against realistic prose rather than isolated bullets.
The document is also intentionally longer than a minimal annex because defense cybersecurity appendices often carry contextual rationale, role expectations, phased implementation guidance, and program-specific restrictions alongside the core mandatory requirements.

## 2. Access Control and Identity Assurance
This section combines specific control language with rationale, implementation notes, and governance context because operational cybersecurity appendices in defense programs rarely appear as a bare matrix. The additional prose is useful for testing chunking and surrounding context while preserving clearly numbered mandatory statements.
- **2.1.1 RC-CYB-01.** Privileged administrators must authenticate through the approved mission identity provider using phishing-resistant multifactor credentials.
  Discussion: RC-CYB-01 exists to reinforce continuity between accreditation evidence, operational support, and sustainment execution. The exact implementation approach can vary so long as the required outcome is preserved and traceable.
- **2.1.2 RC-CYB-02.** Shared maintenance accounts must not be used for routine administration of the fielded relay node.
  Discussion: RC-CYB-02 exists to reinforce continuity between accreditation evidence, operational support, and sustainment execution. The exact implementation approach can vary so long as the required outcome is preserved and traceable.
- **2.1.3 RC-CYB-03.** All support personnel are required to complete role-specific access recertification every 180 days.
  Discussion: RC-CYB-03 exists to reinforce continuity between accreditation evidence, operational support, and sustainment execution. The exact implementation approach can vary so long as the required outcome is preserved and traceable.
- **2.1.4 RC-CYB-04.** Field support laptops may not store reusable administrative passwords in browser caches or unmanaged key stores.
  Discussion: RC-CYB-04 exists to reinforce continuity between accreditation evidence, operational support, and sustainment execution. The exact implementation approach can vary so long as the required outcome is preserved and traceable.
- **2.1.5 RC-CYB-05.** The mission support enclave shall present operators with a lockout warning before a credential reaches its expiration threshold.
  Discussion: RC-CYB-05 exists to reinforce continuity between accreditation evidence, operational support, and sustainment execution. The exact implementation approach can vary so long as the required outcome is preserved and traceable.
- **2.1.6 RC-CYB-06.** Break-glass credentials are prohibited from routine operational use and are required to be escrowed in the sealed emergency access package.
  Discussion: RC-CYB-06 exists to reinforce continuity between accreditation evidence, operational support, and sustainment execution. The exact implementation approach can vary so long as the required outcome is preserved and traceable.
- **2.1.7 RC-CYB-07.** Role assignments are required to be reviewed whenever a user changes operational function, duty location, or employing organization.
  Discussion: RC-CYB-07 exists to reinforce continuity between accreditation evidence, operational support, and sustainment execution. The exact implementation approach can vary so long as the required outcome is preserved and traceable.
- **2.1.8 RC-CYB-08.** Temporary administrator sessions must not remain active after completion of the approved maintenance window.
  Discussion: RC-CYB-08 exists to reinforce continuity between accreditation evidence, operational support, and sustainment execution. The exact implementation approach can vary so long as the required outcome is preserved and traceable.
Background: operators and maintainers frequently request shortcuts during compressed fielding timelines, but those requests should be evaluated as risk-management inputs rather than automatic policy changes. The appendix intentionally keeps this kind of non-binding prose near the mandatory language so extraction quality can be judged against realistic clutter.

## 3. Software Integrity and Supply Chain
This section combines specific control language with rationale, implementation notes, and governance context because operational cybersecurity appendices in defense programs rarely appear as a bare matrix. The additional prose is useful for testing chunking and surrounding context while preserving clearly numbered mandatory statements.
- **3.2.1 RC-CYB-09.** Delivered software packages must include provenance metadata sufficient to identify build pipeline, source revision, and signing identity.
  Discussion: RC-CYB-09 exists to reinforce continuity between accreditation evidence, operational support, and sustainment execution. The exact implementation approach can vary so long as the required outcome is preserved and traceable.
- **3.2.2 RC-CYB-10.** Unsigned container images must not be loaded into the depot integration environment.
  Discussion: RC-CYB-10 exists to reinforce continuity between accreditation evidence, operational support, and sustainment execution. The exact implementation approach can vary so long as the required outcome is preserved and traceable.
- **3.2.3 RC-CYB-11.** Third-party software components are required to be listed in the delivered software bill of materials together with version and license information.
  Discussion: RC-CYB-11 exists to reinforce continuity between accreditation evidence, operational support, and sustainment execution. The exact implementation approach can vary so long as the required outcome is preserved and traceable.
- **3.2.4 RC-CYB-12.** Components with known critical vulnerabilities may not remain in an operational baseline once an approved remediation package is available.
  Discussion: RC-CYB-12 exists to reinforce continuity between accreditation evidence, operational support, and sustainment execution. The exact implementation approach can vary so long as the required outcome is preserved and traceable.
- **3.2.5 RC-CYB-13.** The contractor shall identify all supplier dependencies that can affect patch lead time or vulnerability response timing.
  Discussion: RC-CYB-13 exists to reinforce continuity between accreditation evidence, operational support, and sustainment execution. The exact implementation approach can vary so long as the required outcome is preserved and traceable.
- **3.2.6 RC-CYB-14.** Portable media used for software transfer must be scanned in the staging environment before connection to any mission-support system.
  Discussion: RC-CYB-14 exists to reinforce continuity between accreditation evidence, operational support, and sustainment execution. The exact implementation approach can vary so long as the required outcome is preserved and traceable.
- **3.2.7 RC-CYB-15.** Supplier security attestations are required to identify both current controls and any unresolved corrective actions that could affect delivered software.
  Discussion: RC-CYB-15 exists to reinforce continuity between accreditation evidence, operational support, and sustainment execution. The exact implementation approach can vary so long as the required outcome is preserved and traceable.
- **3.2.8 RC-CYB-16.** Development repositories may not rely on public package mirrors during controlled release builds unless those mirrors are routed through the approved inspection gateway.
  Discussion: RC-CYB-16 exists to reinforce continuity between accreditation evidence, operational support, and sustainment execution. The exact implementation approach can vary so long as the required outcome is preserved and traceable.
Background: operators and maintainers frequently request shortcuts during compressed fielding timelines, but those requests should be evaluated as risk-management inputs rather than automatic policy changes. The appendix intentionally keeps this kind of non-binding prose near the mandatory language so extraction quality can be judged against realistic clutter.

## 4. Logging, Monitoring, and Incident Response
This section combines specific control language with rationale, implementation notes, and governance context because operational cybersecurity appendices in defense programs rarely appear as a bare matrix. The additional prose is useful for testing chunking and surrounding context while preserving clearly numbered mandatory statements.
- **4.3.1 RC-CYB-17.** Security-relevant event records are required to retain node identity, user identity, timestamp, action outcome, and configuration state change details.
  Discussion: RC-CYB-17 exists to reinforce continuity between accreditation evidence, operational support, and sustainment execution. The exact implementation approach can vary so long as the required outcome is preserved and traceable.
- **4.3.2 RC-CYB-18.** Audit records must not be editable by the same role that generated the event.
  Discussion: RC-CYB-18 exists to reinforce continuity between accreditation evidence, operational support, and sustainment execution. The exact implementation approach can vary so long as the required outcome is preserved and traceable.
- **4.3.3 RC-CYB-19.** The fielded node is prohibited from forwarding high-volume debug traffic over constrained mission links during normal operations.
  Discussion: RC-CYB-19 exists to reinforce continuity between accreditation evidence, operational support, and sustainment execution. The exact implementation approach can vary so long as the required outcome is preserved and traceable.
- **4.3.4 RC-CYB-20.** Incident triage teams are required to preserve volatile evidence before initiating a restore action when mission conditions allow.
  Discussion: RC-CYB-20 exists to reinforce continuity between accreditation evidence, operational support, and sustainment execution. The exact implementation approach can vary so long as the required outcome is preserved and traceable.
- **4.3.5 RC-CYB-21.** The relay node shall preserve locally queued cybersecurity events for upload after backhaul restoration.
  Discussion: RC-CYB-21 exists to reinforce continuity between accreditation evidence, operational support, and sustainment execution. The exact implementation approach can vary so long as the required outcome is preserved and traceable.
- **4.3.6 RC-CYB-22.** Maintenance users may not disable audit generation in order to reduce storage consumption during expeditionary operations.
  Discussion: RC-CYB-22 exists to reinforce continuity between accreditation evidence, operational support, and sustainment execution. The exact implementation approach can vary so long as the required outcome is preserved and traceable.
- **4.3.7 RC-CYB-23.** Incident tickets are required to record whether mission impact, data exposure, or assurance degradation occurred during the observed event.
  Discussion: RC-CYB-23 exists to reinforce continuity between accreditation evidence, operational support, and sustainment execution. The exact implementation approach can vary so long as the required outcome is preserved and traceable.
- **4.3.8 RC-CYB-24.** Forensic exports must not omit hash values, baseline identifiers, or operator-observed symptoms that can affect reconstruction of the event timeline.
  Discussion: RC-CYB-24 exists to reinforce continuity between accreditation evidence, operational support, and sustainment execution. The exact implementation approach can vary so long as the required outcome is preserved and traceable.
Background: operators and maintainers frequently request shortcuts during compressed fielding timelines, but those requests should be evaluated as risk-management inputs rather than automatic policy changes. The appendix intentionally keeps this kind of non-binding prose near the mandatory language so extraction quality can be judged against realistic clutter.

## 5. Governance, Accreditation, and Sustainment Assurance
This section combines specific control language with rationale, implementation notes, and governance context because operational cybersecurity appendices in defense programs rarely appear as a bare matrix. The additional prose is useful for testing chunking and surrounding context while preserving clearly numbered mandatory statements.
- **5.4.1 RC-CYB-25.** Accreditation support packages are required to identify residual risk assumptions that depend on external enclaves or shared services.
  Discussion: RC-CYB-25 exists to reinforce continuity between accreditation evidence, operational support, and sustainment execution. The exact implementation approach can vary so long as the required outcome is preserved and traceable.
- **5.4.2 RC-CYB-26.** Waiver requests must not be used to defer remediation of a known exploit path when an approved corrective action already exists.
  Discussion: RC-CYB-26 exists to reinforce continuity between accreditation evidence, operational support, and sustainment execution. The exact implementation approach can vary so long as the required outcome is preserved and traceable.
- **5.4.3 RC-CYB-27.** The program shall maintain a living cyber dependency map for fielded hardware, software, and support-tool components.
  Discussion: RC-CYB-27 exists to reinforce continuity between accreditation evidence, operational support, and sustainment execution. The exact implementation approach can vary so long as the required outcome is preserved and traceable.
- **5.4.4 RC-CYB-28.** Sustainment organizations are required to document how emergency updates are reviewed, approved, distributed, and verified.
  Discussion: RC-CYB-28 exists to reinforce continuity between accreditation evidence, operational support, and sustainment execution. The exact implementation approach can vary so long as the required outcome is preserved and traceable.
- **5.4.5 RC-CYB-29.** Evidence repositories may not store exported mission data longer than the retention period approved for the associated investigation or audit purpose.
  Discussion: RC-CYB-29 exists to reinforce continuity between accreditation evidence, operational support, and sustainment execution. The exact implementation approach can vary so long as the required outcome is preserved and traceable.
- **5.4.6 RC-CYB-30.** Mission rehearsal images must be clearly labeled and are required to remain segregated from operational release candidates.
  Discussion: RC-CYB-30 exists to reinforce continuity between accreditation evidence, operational support, and sustainment execution. The exact implementation approach can vary so long as the required outcome is preserved and traceable.
Background: operators and maintainers frequently request shortcuts during compressed fielding timelines, but those requests should be evaluated as risk-management inputs rather than automatic policy changes. The appendix intentionally keeps this kind of non-binding prose near the mandatory language so extraction quality can be judged against realistic clutter.

## 6. Example Scenarios and Non-Binding Notes
Example implementation details are illustrative only and may vary so long as the mandatory outcomes above remain satisfied. A maintainer may prefer local caching of diagnostic bundles during austere operations. That approach can be acceptable only when the cache is managed under the same assurance controls described above. Likewise, a fielding team may decide to phase some evidence uploads after operations resume, but the timing, preservation, and chain-of-custody expectations described above still govern the process.
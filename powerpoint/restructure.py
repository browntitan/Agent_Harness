"""
Restructure the leadership deck:
1. Reorder slides in presentation.xml
2. Create missing notesSlide files for slides 2-6, 10, 12
3. Update Content_Types.xml for new notes
"""
import os
import re
import random

BASE = "/sessions/gifted-focused-ride/mnt/agentic_chatbot_v3/powerpoint/unpacked_v2"

# ── Step 1: Reorder slides ──────────────────────────────────────
# Map: old slide position (1-indexed) -> (sldId, rId)
CURRENT_ORDER = [
    (256, "rId2"),   # S1  Title
    (386, "rId3"),   # S2  What Is an Agent Harness?
    (387, "rId4"),   # S3  Why LLMs Alone Aren't Enough
    (388, "rId5"),   # S4  Anatomy of an Agent Harness
    (389, "rId6"),   # S5  The ReAct Pattern
    (390, "rId7"),   # S6  Agent Modes
    (257, "rId8"),   # S7  What Does This System Do?
    (258, "rId9"),   # S8  The 30-Second Mental Model
    (392, "rId10"),  # S9  Our Agent Harness at a Glance
    (391, "rId11"),  # S10 Tools & RAG
    (394, "rId12"),  # S11 API Surface
    (393, "rId13"),  # S12 New Subsystems
    (270, "rId14"),  # S13 Three-Layer Runtime
    (271, "rId15"),  # S14 RuntimeService
    (279, "rId16"),  # S15 The Router
    (285, "rId17"),  # S16 All 11 Agent Roles
    (286, "rId18"),  # S17 Agent Mode Execution
    (311, "rId19"),  # S18 Agent Capabilities Matrix
    (288, "rId20"),  # S19 Execution Flow header
    (289, "rId21"),  # S20 Flow 1: Basic
    (290, "rId22"),  # S21 Flow 2: Data Analyst
    (291, "rId23"),  # S22 Flow 3: RAG Worker
    (292, "rId24"),  # S23 Flow 4: Coordinator
    (293, "rId25"),  # S24 Flow 5: Memory
    (294, "rId26"),  # S25 Routing Decision Tree
    (313, "rId27"),  # S26 RAG Pipeline
    (396, "rId28"),  # S27 Authorization
    (397, "rId29"),  # S28 Documents
    (398, "rId30"),  # S29 Skills Architecture
    (399, "rId31"),  # S30 Provider System
    (400, "rId32"),  # S31 MCP Security
    (401, "rId33"),  # S32 Task Plan & Artifact
    (402, "rId34"),  # S33 Storage & Blob
    (403, "rId35"),  # S34 Graph Query Methods
    (404, "rId36"),  # S35 Configuration
    (405, "rId37"),  # S36 Deep RAG Verification
    (406, "rId38"),  # S37 Persistence, Observability
    (260, "rId39"),  # S38 Top-Level Repo Map
    (261, "rId40"),  # S39 Frontend
    (262, "rId41"),  # S40 Control Panel
    (264, "rId42"),  # S41 data/ Directory
    (356, "rId43"),  # S42 GraphRAG
    (357, "rId44"),  # S43 Deep RAG Adaptive
    (358, "rId45"),  # S44 Router Feedback Loop
    (359, "rId46"),  # S45 New Infrastructure
    (360, "rId47"),  # S46 Agentic Harness Improvements
    (354, "rId48"),  # S47 Feature Roadmap
    (380, "rId49"),  # S48 Memory Gaps
    (381, "rId50"),  # S49 Agent Routing Gaps
    (382, "rId51"),  # S50 Tools Gaps
    (383, "rId52"),  # S51 Skills Governance Gaps
    (384, "rId53"),  # S52 Observability Gaps
    (385, "rId54"),  # S53 Enterprise Readiness Matrix
]

# New order by old slide index (0-indexed)
# Section 1: Foundations (1-6) -> old 0,1,2,3,4,5
# Section 2: System Overview (7-10) -> old 6,7,8,10
# Section 3: Runtime (11-14) -> old 12,13,29,34
# Section 4: Routing (15-17) -> old 14,24,43
# Section 5: Agents (18-20) -> old 15,16,17
# Section 6: Tools/MCP/Workers (21-24) -> old 11,9,30,31
# Section 7: RAG (25-27) -> old 25,42,35
# Section 8: GraphRAG (28-29) -> old 41,33
# Section 9: Flows (30-35) -> old 18,19,20,21,22,23
# Section 10: Subsystems (36-41) -> old 26,27,28,32,36,44
# Section 11: Codebase (42-45) -> old 37,38,39,40
# Section 12: Roadmap (46-53) -> old 45,46,47,48,49,50,51,52

NEW_ORDER_INDICES = [
    0,1,2,3,4,5,       # S1-6: Foundations
    6,7,8, 10,          # S7-10: System Overview (old S11=API moved here)
    12,13, 29, 34,      # S11-14: Runtime (old S13,S14,S30,S35)
    14, 24, 43,         # S15-17: Routing (old S15,S25,S44)
    15,16,17,           # S18-20: Agents (old S16,S17,S18)
    11, 9, 30, 31,      # S21-24: Tools/MCP/Workers (old S12,S10,S31,S32)
    25, 42, 35,         # S25-27: RAG (old S26,S43,S36)
    41, 33,             # S28-29: GraphRAG (old S42,S34)
    18,19,20,21,22,23,  # S30-35: Flows (old S19-S24)
    26,27,28, 32, 36, 44, # S36-41: Subsystems (old S27,S28,S29,S33,S37,S45)
    37,38,39,40,        # S42-45: Codebase/Frontend (old S38-S41)
    45,46,47,48,49,50,51,52  # S46-53: Roadmap/Gaps (old S46-S53)
]

assert len(NEW_ORDER_INDICES) == 53, f"Expected 53, got {len(NEW_ORDER_INDICES)}"
assert len(set(NEW_ORDER_INDICES)) == 53, "Duplicate indices!"

new_sld_list = []
for idx in NEW_ORDER_INDICES:
    sid, rid = CURRENT_ORDER[idx]
    new_sld_list.append(f'      <p:sldId id="{sid}" r:id="{rid}"/>')

new_sld_xml = "\n".join(new_sld_list)

# Read and replace in presentation.xml
pres_path = os.path.join(BASE, "ppt/presentation.xml")
with open(pres_path, "r") as f:
    content = f.read()

# Find and replace the sldIdLst block
old_pattern = re.compile(r'(<p:sldIdLst>)\s*(.*?)\s*(</p:sldIdLst>)', re.DOTALL)
match = old_pattern.search(content)
if match:
    new_block = f"{match.group(1)}\n{new_sld_xml}\n    {match.group(3)}"
    content = content[:match.start()] + new_block + content[match.end():]
    with open(pres_path, "w") as f:
        f.write(content)
    print("✓ Reordered slides in presentation.xml")
else:
    print("✗ Could not find sldIdLst!")

# ── Step 2: Create missing notes files ──────────────────────────
# Slides without notes: 2,3,4,5,6,10,12 (old numbering)
# These are slide2.xml through slide6.xml, slide10.xml, slide12.xml
SLIDES_WITHOUT_NOTES = [2, 3, 4, 5, 6, 10, 12]

# Next available notesSlide number
existing_notes = [int(re.search(r'(\d+)', f).group(1))
                  for f in os.listdir(os.path.join(BASE, "ppt/notesSlides"))
                  if f.startswith("notesSlide") and f.endswith(".xml")]
next_num = max(existing_notes) + 1

notes_template = '''<?xml version="1.0" encoding="utf-8"?>
<p:notes xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:spTree>
      <p:nvGrpSpPr>
        <p:cNvPr id="1" name=""/>
        <p:cNvGrpSpPr/>
        <p:nvPr/>
      </p:nvGrpSpPr>
      <p:grpSpPr>
        <a:xfrm>
          <a:off x="0" y="0"/>
          <a:ext cx="0" cy="0"/>
          <a:chOff x="0" y="0"/>
          <a:chExt cx="0" cy="0"/>
        </a:xfrm>
      </p:grpSpPr>
      <p:sp>
        <p:nvSpPr>
          <p:cNvPr id="2" name="Slide Image Placeholder 1"/>
          <p:cNvSpPr>
            <a:spLocks noGrp="1" noRot="1" noChangeAspect="1"/>
          </p:cNvSpPr>
          <p:nvPr>
            <p:ph type="sldImg"/>
          </p:nvPr>
        </p:nvSpPr>
        <p:spPr/>
      </p:sp>
      <p:sp>
        <p:nvSpPr>
          <p:cNvPr id="3" name="Notes Placeholder 2"/>
          <p:cNvSpPr>
            <a:spLocks noGrp="1"/>
          </p:cNvSpPr>
          <p:nvPr>
            <p:ph type="body" idx="1"/>
          </p:nvPr>
        </p:nvSpPr>
        <p:spPr/>
        <p:txBody>
          <a:bodyPr/>
          <a:lstStyle/>
          <a:p>
            <a:r>
              <a:rPr lang="en-US" dirty="0"/>
              <a:t>PLACEHOLDER</a:t>
            </a:r>
          </a:p>
        </p:txBody>
      </p:sp>
      <p:sp>
        <p:nvSpPr>
          <p:cNvPr id="4" name="Slide Number Placeholder 3"/>
          <p:cNvSpPr>
            <a:spLocks noGrp="1"/>
          </p:cNvSpPr>
          <p:nvPr>
            <p:ph type="sldNum" sz="quarter" idx="10"/>
          </p:nvPr>
        </p:nvSpPr>
        <p:spPr/>
        <p:txBody>
          <a:bodyPr/>
          <a:lstStyle/>
          <a:p>
            <a:fld id="{{F7021451-1387-4CA6-816F-3879F97B5CBC}}" type="slidenum">
              <a:rPr lang="en-US"/>
              <a:t>0</a:t>
            </a:fld>
            <a:endParaRPr lang="en-US"/>
          </a:p>
        </p:txBody>
      </p:sp>
    </p:spTree>
    <p:extLst>
      <p:ext uri="{{BB962C8B-B14F-4D97-AF65-F5344CB8AC3E}}">
        <p14:creationId xmlns:p14="http://schemas.microsoft.com/office/powerpoint/2010/main" val="{creation_id}"/>
      </p:ext>
    </p:extLst>
  </p:cSld>
  <p:clrMapOvr>
    <a:masterClrMapping/>
  </p:clrMapOvr>
</p:notes>'''

rels_template = '''<?xml version="1.0" encoding="utf-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="../slides/slide{slide_num}.xml"/>
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesMaster" Target="../notesMasters/notesMaster1.xml"/>
</Relationships>'''

# Read Content_Types.xml
ct_path = os.path.join(BASE, "[Content_Types].xml")
with open(ct_path, "r") as f:
    ct_content = f.read()

for slide_num in SLIDES_WITHOUT_NOTES:
    notes_num = next_num
    next_num += 1
    
    # Create notesSlide XML
    notes_xml = notes_template.format(creation_id=random.randint(100000000, 999999999))
    notes_file = os.path.join(BASE, f"ppt/notesSlides/notesSlide{notes_num}.xml")
    with open(notes_file, "w") as f:
        f.write(notes_xml)
    
    # Create rels file
    rels_xml = rels_template.format(slide_num=slide_num)
    rels_dir = os.path.join(BASE, "ppt/notesSlides/_rels")
    rels_file = os.path.join(rels_dir, f"notesSlide{notes_num}.xml.rels")
    with open(rels_file, "w") as f:
        f.write(rels_xml)
    
    # Add relationship to slide's rels file
    slide_rels_path = os.path.join(BASE, f"ppt/slides/_rels/slide{slide_num}.xml.rels")
    with open(slide_rels_path, "r") as f:
        slide_rels = f.read()
    
    # Find max rId in the slide rels
    rids = re.findall(r'Id="rId(\d+)"', slide_rels)
    max_rid = max(int(r) for r in rids) if rids else 0
    new_rid = f"rId{max_rid + 1}"
    
    # Add notes relationship before closing tag
    new_rel = f'  <Relationship Id="{new_rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesSlide" Target="../notesSlides/notesSlide{notes_num}.xml"/>'
    slide_rels = slide_rels.replace("</Relationships>", f"{new_rel}\n</Relationships>")
    with open(slide_rels_path, "w") as f:
        f.write(slide_rels)
    
    # Add to Content_Types.xml
    ct_entry = f'  <Override PartName="/ppt/notesSlides/notesSlide{notes_num}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.notesSlide+xml"/>'
    ct_content = ct_content.replace("</Types>", f"{ct_entry}\n</Types>")
    
    print(f"✓ Created notesSlide{notes_num}.xml for slide{slide_num}.xml")

# Write updated Content_Types
with open(ct_path, "w") as f:
    f.write(ct_content)

print(f"\n✓ All {len(SLIDES_WITHOUT_NOTES)} missing notes files created")
print("✓ Content_Types.xml updated")
print("\nDone! Structural changes complete.")

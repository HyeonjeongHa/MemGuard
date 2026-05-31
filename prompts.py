"""
Prompts for the two-call memory extraction pipeline.

Ordering per session:
  Call 1 — PROMPT_A_EXTRACT_RELATE_ROUTE
      Input : conversation_timestamp + new messages
      Output: typed atoms + new-to-new links
      (no existing memories → no distraction)

  Call 2 — PROMPT_B_ASSIGN_OPS_LINKS
      Input : atoms from Call 1 + existing memories + conversation_timestamp
      Output: ADD/UPDATE/SKIP operation per atom + new-to-existing links

Relation labels (unified, type-agnostic):
  supports     — source supports, evidences, or is required by target
  instance_of  — source is a concrete occurrence of the target concept
  derived_from — source stable fact was concluded from / triggered by target experience
  leads_to     — source event directly caused or led to target event
  context_for  — source provides background or context for target
  elaborates   — source adds detail or specificity to target
  contradicts  — source conflicts with or updates target
"""

# ---------------------------------------------------------------------------
# Retrieval-time routing prompts
# ---------------------------------------------------------------------------

QUERY_MEMORY_ROUTE_PROMPT_RELATIONS = """\
You are a memory routing assistant. Given a user query, decide which memory type(s) are most likely to contain the answer.

Memory types:
- semantic: timeless and stable facts about a person or a world
- episodic: time-anchored past/future events or experiences 
- procedural: recurring behaviors or routines 

Query: {{user_query}}

Return a JSON array of the relevant memory type(s), ordered from most to least relevant.
Example: ["episodic", "semantic"]
Return ONLY the JSON array.\
"""


QUERY_WEIGHT_PROMPT_RELATIONS = """\
You are a memory routing assistant. Given a user query, assign a confidence weight (0.0–1.0) to each memory type that may contain the answer. Weights must sum to 1.0.

Memory types:
- semantic: timeless and stable facts about a person or a world
- episodic: time-anchored past/future events or experiences 
- procedural: recurring behaviors or routines 

Query: {{user_query}}

Return ONLY a JSON object:
{"weights": {"semantic": <float>, "episodic": <float>, "procedural": <float>}}\
"""

# ---------------------------------------------------------------------------
# Call 1 — Extract, Relate, Route
# ---------------------------------------------------------------------------

PROMPT_A_EXTRACT_RELATE_ROUTE = """\
You are an expert knowledge extraction, relationship analysis, and routing system. Your job is to:
1. Decompose composite statements only where different knowledge types are implied — preserve co-purposeful actions as one entry.
2. Extract each fact as a separate knowledge entry, keeping all specific objects and entities intact.
3. Identify relationships between knowledge entries.
4. Route each entry to the correct memory type.

### CONTEXT
Conversation Timestamp: {{conversation_timestamp}}

New Messages:
{{messages}}

---

### PHASE 1: EXTRACT
Scan the conversation for every useful piece of knowledge. Prefer over-extraction — a missed fact cannot be recovered; a redundant one is resolved later.

**What to extract** (capture all that apply):
- Identity & demographics: name, age, occupation, location, education, background
- Personality & traits: self-described style, emotional patterns, decision tendencies, communication style
- Preferences & tastes: food, entertainment, tech tools, travel, aesthetics, communication channels
- Values & beliefs: ethical principles, priorities, worldview, attitudes toward money/work/life
- Goals & plans: short-term tasks, long-term ambitions, learning objectives, financial/health goals
- Constraints & challenges: budget limits, active problems, fears, non-negotiables, frustrations
- Social relationships: family, partner, friends, colleagues, pets — with names and key facts
- Health & wellbeing: conditions, medications, allergies, exercise habits, diet, sleep
- Possessions & resources: devices, vehicles, subscriptions, financial resources, creative tools
- Skills & expertise: professional skills, years of experience, languages, areas actively learning
- Projects & work: active projects (name, goal, status, collaborators), responsibilities, affiliations
- Routines & habits: recurring daily/weekly behaviors, work habits, financial habits, creative habits
- Commitments & obligations: promises to named people, appointments with dates, ongoing duties
- Life events: past/future occurrences, milestones, first-time experiences, reactions
- Opinions & evaluations: ratings, recommendations, positive/negative/mixed reactions to products/people
- Emotional states & reactions: expressed feelings tied to specific events or situations; emotional tone toward named people or places; significant emotional turning points volunteered by the speaker
- Current life situation: life phase, recent major transitions, environmental/seasonal context
- Domain knowledge: custom vocabulary, frameworks, named systems or tools they personally built

**Decomposition — split composite statements before extracting:**
Scan every statement for facts that belong to different memory types bundled together. Split only when types diverge; keep co-purposeful actions together.

| Split trigger | How to split |
|---|---|
| Past event + derived stable belief or trait | Episodic entry for the event; semantic entry for the belief |
| Occurrence + timeless motivational/emotional explanation | Separate entry for the occurrence; separate entry for the stable motivation |
| Stable preference + the one-time event that revealed it | Episodic for the event; semantic for the preference |
| Two or more subjects sharing one predicate | One entry per subject |

**Do NOT split when:**
- Multiple actions share the same subject, context, and type (e.g., "will do A and B for C" → one entry)
- Multiple reasons/effects all belong to the same type → merge into one entry

Example — *"Melanie painted a lake sunrise last year and finds painting a fun way to express herself, get creative, and relax."*
→ atom 0: Melanie painted a lake sunrise (time-anchored occurrence)
→ atom 1: Melanie uses painting to express herself, get creative, and relax (stable multi-purpose belief — kept together since all purposes are semantic)

Counter-example — *"I'm going to launch a website and run ads for my limited edition hoodie line."*
→ ONE episodic entry (same type, same subject, same context — do not split)

**What to skip:**
- Common public knowledge (e.g. "Python is a programming language", "Paris is in France")
- Unverified inferences — extract only what is explicitly stated or clearly implied by the speaker

**Completeness rules:**
- Never omit named objects, products, places, quantities, or people from an entry
- Preserve exact names, numbers, dates, units, and quoted phrases
- Never bundle a time-anchored event and the stable belief it implies in one entry
- Never mix information about different people in the same entry
- If uncertain or ambiguous, set "uncertain": true

**Time normalization:**
- `time` records when the event actually occurred (YYYY-MM-DD, YYYY-MM, or YYYY) — NOT the conversation timestamp
- Resolve relative expressions ("next month", "last year") to absolute dates using the Conversation Timestamp
- In `details`, include both the resolved event time and the conversation timestamp so the entry is fully self-contained.
  - Format: `<resolved_time>: <fact> (mentioned on <conversation_timestamp>)`
  - Example: conversation timestamp 2022-03-27, speaker says "I've been playing drums for a month":
      → resolved event start: 2022-02
      → `time`: "2022-02"
      → `details`: "John has been playing drums for a month (mentioned on 2022-03-27); he described it as tough but fun."
- If the day-of-week is needed but not known (e.g. "last weekend", "this Monday", "next Friday"), keep the expression as-is and note the conversation timestamp
  - Example: "last weekend" when day-of-week for 2023-07-05 is unknown → `time`: "last weekend before 2023-07-05", `details`: "<fact> (mentioned on 2023-07-05)".
- If no time is mentioned, use the Conversation Timestamp for `time`

---

### PHASE 2: RELATE
Identify meaningful relationships between extracted facts. Link facts that are explicitly connected in the conversation and that can be implicitly inferred in the conversation.

Use exactly these relation labels:

| Label        | Meaning                                                             |
|---|---|
| `supports`   | source supports, evidences, or is required by / informed by target  |
| `instance_of`| source is a concrete occurrence of the target concept               |
| `derived_from` | source stable fact was concluded from or triggered by target experience |
| `leads_to`   | source event directly caused or led to target event                 |
| `context_for`| source provides background or context for target                    |
| `elaborates` | source adds detail or specificity to target                         |
| `contradicts`| source conflicts with or updates target                             |

Rules:
- Each link is directional: source → target follows the relation's meaning
- One atom may appear in multiple links
- If no relationships exist, output "links": []

---

### PHASE 3: ROUTE
Assign exactly one type to every extracted atom:

- **semantic**: timeless and stable facts about a person or a world
- **episodic**: time-anchored past/future events or experiences 
- **procedural**: recurring behaviors or routines 

**Guardrails:**
- Past, long-time ago experience is anchored to time → episodic, NOT semantic
- A one-time description of how to do something → episodic, NOT procedural
- Procedural is ONLY for behaviors the person performs on a recurring basis

---

### OUTPUT FORMAT
Return a single JSON object:

{
  "atoms": [
    {
      "id": 0,
      "type": "semantic" | "episodic" | "procedural",
      "title": "Short label (≤10 words)",
      "details": "Full details including resolved event time and conversation timestamp where applicable",
      "time": "YYYY-MM-DD|YYYY-MM|YYYY",
      "uncertain": true|false
    }
  ],
  "links": [
    {
      "source": 0,
      "target": 1,
      "relation": "<relation_label>",
      "reasoning": "<one sentence grounded in the conversation>"
    }
  ]
}

### CRITICAL RULES:
1. Return ONLY the JSON object — no preamble, explanation, or trailing text
2. Every extracted fact must appear as exactly one atom
3. Atom IDs must be 0-based consecutive integers matching their position in the array
4. Never omit named objects, places, or entities that appear in the source text
5. "title" must be unique and self-explanatory without surrounding context\
"""


# ---------------------------------------------------------------------------
# Call 1.5 (optional) — Self-Check Extraction
# ---------------------------------------------------------------------------

PROMPT_C_SELF_CHECK_EXTRACTION = """\
You are a memory extraction auditor. A first-pass extraction has already been run on the conversation below. Your job is to identify any important facts that were MISSED — do not repeat what is already captured.

### CONTEXT
Conversation Timestamp: {{conversation_timestamp}}

New Messages:
{{messages}}

---

### Already Extracted Atoms (do NOT duplicate these):
{{existing_atoms_json}}

---

### TASK
Carefully re-read the conversation and list any important facts NOT yet captured above.

**Extraction Rules:**
- Only report genuinely new facts — skip anything already covered by an existing atom
- Skip common public knowledge and unverified inferences
- Preserve exact names, numbers, dates, and entities
- Split composite facts that span different types (episodic event vs. semantic belief)
- Use `time` for when the event occurred (YYYY-MM-DD, YYYY-MM, or YYYY); resolve relative expressions using the Conversation Timestamp
- New atom IDs must start at {{next_id}} and increment consecutively

**Relation Rules:**
- Links may connect new additional atoms to each other OR to existing atoms (using their existing IDs)
- Use exactly these relation labels:
  | Label        | Meaning                                                             |
  |---|---|
  | `supports`   | source supports, evidences, or is required by / informed by target  |
  | `instance_of`| source is a concrete occurrence of the target concept               |
  | `derived_from` | source stable fact was concluded from or triggered by target experience |
  | `leads_to`   | source event directly caused or led to target event                 |
  | `context_for`| source provides background or context for target                    |
  | `elaborates` | source adds detail or specificity to target                         |
  | `contradicts`| source conflicts with or updates target                             |

**Routing Rules:**
Assign exactly one type to every extracted atom:

- **semantic**: timeless and stable facts about a person or a world
- **episodic**: time-anchored past/future events or experiences 
- **procedural**: recurring behaviors or routines 

If nothing was missed, return `{"additional_atoms": [], "additional_links": []}`.

---

### OUTPUT FORMAT
{
  "additional_atoms": [
    {
      "id": {{next_id}},
      "type": "semantic" | "episodic" | "procedural",
      "title": "Short label (≤10 words)",
      "details": "Full details including resolved event time and conversation timestamp",
      "time": "YYYY-MM-DD|YYYY-MM|YYYY",
      "uncertain": true|false
    }
  ],
  "additional_links": [
    {
      "source": <atom_id>,
      "target": <atom_id>,
      "relation": "<relation_label>",
      "reasoning": "<one sentence grounded in the conversation>"
    }
  ]
}

### CRITICAL RULES:
1. Return ONLY the JSON object — no preamble, explanation, or trailing text
2. New atom IDs must start at {{next_id}} — never reuse existing atom IDs
3. Never omit named objects, places, or entities from atom details\
"""

# ---------------------------------------------------------------------------
# Call 2 — Assign Operations + Existing Links
# ---------------------------------------------------------------------------

PROMPT_B_ASSIGN_OPS_LINKS = """\
You are a memory operation assignment system. Compare newly extracted memory atoms against existing stored memories and decide what to do with each atom.

### CONTEXT
Conversation Timestamp: {{conversation_timestamp}}

Existing Semantic Memories (compare ONLY for semantic atoms):
{{existing_semantic_memories}}

Existing Episodic Memories (compare ONLY for episodic atoms):
{{existing_episodic_memories}}

Existing Procedural Memories (compare ONLY for procedural atoms):
{{existing_procedural_memories}}

New Memory Atoms:
{{atoms_json}}

---

### PHASE 4: ASSIGN OPERATIONS
For every atom, compare it against existing memories OF THE SAME TYPE and assign exactly one action:

- **ADD**: No existing memory covers this fact. Create a new entry.
- **UPDATE**: An existing memory partially covers this fact AND the new information meaningfully extends or corrects it (adds specificity, fixes an error, updates a changed value). Include `old_memory_id`.
- **SKIP**: The fact is already fully captured by an existing memory. Include `existing_id` (the ID of the overlapping memory). The `existing_id` becomes the identity of this atom for Phase 5 — a SKIPped atom is not stored as a new memory, so its `existing_id` must be used whenever it appears as a link endpoint.

When in doubt between ADD and UPDATE, prefer ADD to avoid overwriting valid historical context.

Compare:
- semantic atoms → against existing_semantic_memories
- episodic atoms → against existing_episodic_memories
- procedural atoms → against existing_procedural_memories

---

### PHASE 5: EXISTING LINKS
Links between new atoms were already captured in Phase 2 (RELATE). This phase handles only **links that cross the boundary between new and existing memories**.

Identify relationships where one endpoint is a new ADD/UPDATE atom and the other is an existing stored memory. This covers:
- A new ADD/UPDATE atom that elaborates, contradicts, or provides context for an existing memory
- A SKIPped atom whose `existing_id` is meaningfully connected to another ADD/UPDATE atom — use the SKIP's `existing_id` as the endpoint, not the atom index

**Do NOT create links where both endpoints are new atoms** — those belong to Phase 2 and are already recorded.

Use exactly these relation labels:

| Label        | Meaning                                                             |
|---|---|
| `supports`   | source supports, evidences, or is required by / informed by target  |
| `instance_of`| source is a concrete occurrence of the target concept               |
| `derived_from` | source stable fact was concluded from or triggered by target experience |
| `leads_to`   | source event directly caused or led to target event                 |
| `context_for`| source provides background or context for target                    |
| `elaborates` | source adds detail or specificity to target                         |
| `contradicts`| source conflicts with or updates target                             |

Each link needs exactly one source and one target. Choose the correct field:
- ADD/UPDATE atom as endpoint: use `source_atom` or `target_atom` (integer atom id)
- Existing memory as endpoint (including a SKIP's `existing_id`): use `source_existing_id` or `target_existing_id` (memory id string)

**Do NOT use `source_atom`/`target_atom` for SKIPped atoms** — they are not stored as new memories. Always reference them via `source_existing_id`/`target_existing_id`.

Only create links explicitly grounded in the conversation. If none exist, output "existing_links": [].

---

### OUTPUT FORMAT
Return a single JSON object:

{
  "operations": [
    {"atom_id": 0, "action": "ADD"},
    {"atom_id": 1, "action": "UPDATE", "old_memory_id": "<existing_memory_id>"},
    {"atom_id": 2, "action": "SKIP", "existing_id": "<existing_memory_id>"}
  ],
  "existing_links": [
    {
      "source_atom": 0,
      "target_existing_id": "<existing_memory_id>",
      "relation": "<relation_label>",
      "reasoning": "<one sentence grounded in the conversation>"
    },
    {
      "source_existing_id": "<existing_memory_id>",
      "target_atom": 1,
      "relation": "<relation_label>",
      "reasoning": "<one sentence>"
    },
    {
      "source_existing_id": "<skip_atom_existing_id>",
      "target_atom": 2,
      "relation": "<relation_label>",
      "reasoning": "<atom 2 is new ADD/UPDATE; the SKIP atom is referenced via its existing_id>"
    }
  ]
}

### CRITICAL RULES:
1. Return ONLY the JSON object — no preamble, explanation, or trailing text
2. Every atom must appear in exactly one operation
3. SKIP operations must include `existing_id`
4. UPDATE operations must include `old_memory_id`\
"""

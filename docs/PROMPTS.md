# Prompt Templates

The pipeline ships with five Jinja2 prompt templates, defined in `src/prompts/templates.yaml` and loaded by `src/prompts/renderer.py`.

Each template produces two messages for the chat API:
- a **system** message that sets the LLM's role, and
- a **user** message that contains the triple and asks for a score.

## Template Reference

### 1. `minimal` — Minimal (Labels Only)

**Description:** Uses only entity labels, no descriptions. Shortest and cheapest prompt.

**System prompt:**
```
You are a knowledge graph expert. Respond only in JSON.
```

**User prompt template:**
```
Rate the plausibility of this knowledge graph triple on a scale from 0.0 to 1.0.

Head entity: {{ head_label }}
Relation: {{ relation }}
Tail entity: {{ tail_label }}

Reply in JSON: {"score": <float>, "reason": "<string>"}
```

**Example rendered output** for `(Barack Obama, /people/person/nationality, United States of America)`:
```json
[
  {"role": "system", "content": "You are a knowledge graph expert. Respond only in JSON."},
  {"role": "user", "content": "Rate the plausibility of this knowledge graph triple on a scale from 0.0 to 1.0.\n\nHead entity: Barack Obama\nRelation: /people/person/nationality\nTail entity: United States of America\n\nReply in JSON: {\"score\": <float>, \"reason\": \"<string>\"}"}
]
```

---

### 2. `with_descriptions` — With Descriptions

**Description:** Includes entity labels and Wikidata descriptions for richer context.

**System prompt:**
```
You are a knowledge graph expert. Respond only in JSON.
```

**User prompt template:**
```
Rate the plausibility of this knowledge graph triple on a scale from 0.0 to 1.0.

Head entity: {{ head_label }}{% if head_description %} — {{ head_description }}{% endif %}
Relation: {{ relation }}
Tail entity: {{ tail_label }}{% if tail_description %} — {{ tail_description }}{% endif %}

Reply in JSON: {"score": <float>, "reason": "<string>"}
```

**Example rendered output:**
```json
[
  {"role": "system", "content": "You are a knowledge graph expert. Respond only in JSON."},
  {"role": "user", "content": "Rate the plausibility of this knowledge graph triple on a scale from 0.0 to 1.0.\n\nHead entity: Barack Obama — 44th president of the United States\nRelation: /people/person/nationality\nTail entity: United States of America — federal republic in North America\n\nReply in JSON: {\"score\": <float>, \"reason\": \"<string>\"}"}
]
```

---

### 3. `strict_rubric` — Strict Scoring Rubric

**Description:** Provides a detailed rubric for scoring to improve calibration.

**System prompt:**
```
You are a knowledge graph evaluation system. You must respond only in valid JSON.
```

**User prompt template:**
```
Evaluate whether the following knowledge graph triple is factually correct.

Head: {{ head_label }}{% if head_description %} ({{ head_description }}){% endif %}
Relation: {{ relation }}
Tail: {{ tail_label }}{% if tail_description %} ({{ tail_description }}){% endif %}

Scoring rubric:
- 0.9–1.0: Almost certainly true, well-known fact
- 0.6–0.8: Likely true but not certain
- 0.3–0.5: Uncertain, could go either way
- 0.1–0.2: Likely false
- 0.0–0.1: Almost certainly false

Reply in JSON: {"score": <float>, "reason": "<string>"}
```

**Example rendered output:**
```json
[
  {"role": "system", "content": "You are a knowledge graph evaluation system. You must respond only in valid JSON."},
  {"role": "user", "content": "Evaluate whether the following knowledge graph triple is factually correct.\n\nHead: Barack Obama (44th president of the United States)\nRelation: /people/person/nationality\nTail: United States of America (federal republic in North America)\n\nScoring rubric:\n- 0.9–1.0: Almost certainly true, well-known fact\n..."}
]
```

---

### 4. `concise_cot` — Concise Chain-of-Thought

**Description:** Asks for brief reasoning before the score, encouraging step-by-step thinking.

**System prompt:**
```
You are a knowledge graph expert. Think step by step, then give a score. Respond in JSON.
```

**User prompt template:**
```
Is the following triple plausible?

({{ head_label }}, {{ relation }}, {{ tail_label }})

Think briefly, then reply in JSON:
{"reason": "<1-2 sentence reasoning>", "score": <float>}
```

**Example rendered output:**
```json
[
  {"role": "system", "content": "You are a knowledge graph expert. Think step by step, then give a score. Respond in JSON."},
  {"role": "user", "content": "Is the following triple plausible?\n\n(Barack Obama, /people/person/nationality, United States of America)\n\nThink briefly, then reply in JSON:\n{\"reason\": \"<1-2 sentence reasoning>\", \"score\": <float>}"}
]
```

---

### 5. `binary_judge` — Binary Judge

**Description:** Asks for a binary true/false judgment plus confidence. Score is derived from the confidence value.

**System prompt:**
```
You are a factual knowledge evaluator. Respond only in JSON.
```

**User prompt template:**
```
Is this knowledge graph triple true or false?

Head: {{ head_label }}
Relation: {{ relation }}
Tail: {{ tail_label }}

Reply in JSON: {"judgment": "true"|"false", "confidence": <float>}
```

**Example rendered output:**
```json
[
  {"role": "system", "content": "You are a factual knowledge evaluator. Respond only in JSON."},
  {"role": "user", "content": "Is this knowledge graph triple true or false?\n\nHead: Barack Obama\nRelation: /people/person/nationality\nTail: United States of America\n\nReply in JSON: {\"judgment\": \"true\"|\"false\", \"confidence\": <float>}"}
]
```

---

## Adding a New Template

1. Open `src/prompts/templates.yaml`.
2. Add a new top-level key under `templates:` with the following structure:

```yaml
templates:
  my_new_template:
    id: "my_new_template"
    name: "Human-Readable Name"
    description: "Short description of what this template does."
    system: "System prompt text here."
    user: |
      User prompt template here.
      Use {{ head_label }}, {{ relation }}, {{ tail_label }},
      {{ head_description }}, {{ tail_description }} as Jinja2 variables.
      The response must be parseable JSON containing at least a "score" key.
```

3. Add a test case in `tests/test_prompt_renderer.py` to verify rendering.
4. The template is immediately available by its `id` via `TemplateRegistry.get("my_new_template")`.

### Template Variable Reference

| Variable | Type | Description |
|---|---|---|
| `head_label` | `str` | Human-readable label for the head entity |
| `relation` | `str` | Relation string (e.g. `/people/person/nationality`) |
| `tail_label` | `str` | Human-readable label for the tail entity |
| `head_description` | `str` | Wikidata description for the head entity (may be empty) |
| `tail_description` | `str` | Wikidata description for the tail entity (may be empty) |

### Response Format Requirements

The LLM response must be valid JSON. The scorer (`src/models/scorer.py`) extracts the `"score"` field (a float in `[0, 1]`). The `"reason"` field is optional but recommended for debugging. The `binary_judge` template uses `"confidence"` instead of `"score"`.

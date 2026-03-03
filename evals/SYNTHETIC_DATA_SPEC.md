# Synthetic Consultation Data Generator Specification

## Overview

This specification defines the output schema and requirements for generating synthetic consultation datasets for ThemeFinder evaluation.

---

## Output Structure

```
{dataset_name}/
├── inputs/
│   ├── question_part_1/
│   │   ├── question.json
│   │   └── responses.jsonl
│   ├── question_part_2/
│   │   └── ...
│   └── respondents.jsonl
└── outputs/
    └── mapping/
        └── {date}/
            ├── question_part_1/
            │   ├── themes.json
            │   ├── sentiment.jsonl
            │   ├── mapping.jsonl
            │   └── detail_detection.jsonl
            └── question_part_2/
                └── ...
```

---

## Schema Definitions

### 1. Question (`inputs/question_part_N/question.json`)

```json
{
    "question_number": 1,
    "question_text": "What measures should we consider to improve X?",
    "multi_choice_options": ["Support", "Oppose"]  // Optional
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `question_number` | integer | Yes | Sequential question identifier |
| `question_text` | string | Yes | The consultation question |
| `multi_choice_options` | array[string] | No | Pre-defined response options if applicable |

---

### 2. Responses (`inputs/question_part_N/responses.jsonl`)

```json
{"response_id": 1, "response": "I believe we should..."}
{"response_id": 2, "response": "This policy would harm..."}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `response_id` | integer | Yes | Unique identifier (unique within question part) |
| `response` | string | Yes | Free-text response |

---

### 3. Respondents (`inputs/respondents.jsonl`)

```json
{"response_id": 1, "demographic_data": {"region": "England", "age_group": "25-34"}}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `response_id` | integer | Yes | Links to response |
| `demographic_data` | object | Yes | Arbitrary demographic key-value pairs |

---

### 4. Themes (`outputs/mapping/{date}/question_part_N/themes.json`)

```json
[
    {
        "topic_id": "A",
        "topic_label": "Funding Concerns",
        "topic_description": "Concerns about adequate funding for the proposed initiative.",
        "topic": "Funding Concerns: Concerns about adequate funding for the proposed initiative."
    }
]
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `topic_id` | string | Yes | Single letter identifier (A-Z, then AA-AZ if needed) |
| `topic_label` | string | Yes | Short theme name (2-5 words) |
| `topic_description` | string | Yes | Detailed description (1-2 sentences) |
| `topic` | string | Yes | Combined `{label}: {description}` |

**Special Themes (always include):**
- `X`: "None of the Above" - Response doesn't match any theme
- `Y`: "No Reason Given" - Response lacks substantive content

---

### 5. Sentiment (`outputs/mapping/{date}/question_part_N/sentiment.jsonl`)

```json
{"response_id": 1, "position": "AGREE"}
{"response_id": 2, "position": "DISAGREE"}
{"response_id": 3, "position": "UNCLEAR"}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `response_id` | integer | Yes | Links to response |
| `position` | enum | Yes | One of: `AGREE`, `DISAGREE`, `UNCLEAR` |

---

### 6. Mapping (`outputs/mapping/{date}/question_part_N/mapping.jsonl`)

```json
{"response_id": 1, "labels": ["A", "C"], "stances": ["POSITIVE", "NEGATIVE"]}
{"response_id": 2, "labels": ["B"], "stances": ["POSITIVE"]}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `response_id` | integer | Yes | Links to response |
| `labels` | array[string] | Yes | List of topic_ids (1-3 typically) |
| `stances` | array[string] | Yes | Stance per label: `POSITIVE`, `NEGATIVE`, `NEUTRAL` |

**Constraints:**
- `labels` and `stances` arrays must have equal length
- `labels` must reference valid `topic_id` values from themes.json
- Empty `labels` not permitted (use `X` or `Y` for edge cases)

---

### 7. Detail Detection (`outputs/mapping/{date}/question_part_N/detail_detection.jsonl`)

```json
{"response_id": 1, "evidence_rich": "YES"}
{"response_id": 2, "evidence_rich": "NO"}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `response_id` | integer | Yes | Links to response |
| `evidence_rich` | enum | Yes | `YES` if response contains specific evidence/examples, `NO` otherwise |

---

## Generation Parameters

### Required Configuration

| Parameter | Type | Description |
|-----------|------|-------------|
| `topic` | string | Consultation subject (e.g., "housing", "healthcare", "transport") |
| `size` | enum | `XS` (100), `S` (500), `M` (1000), `L` (5000) |
| `n_questions` | integer | Number of question parts (1-5) |

### Optional Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_themes_per_question` | integer | 10-15 | Target theme count |
| `noise_level` | enum | `medium` | `low`, `medium`, `high` - controls typos, off-topic, etc. |
| `position_distribution` | object | `{agree: 0.5, disagree: 0.3, unclear: 0.2}` | Sentiment balance |
| `response_length_distribution` | object | `{short: 0.2, medium: 0.6, long: 0.2}` | Length variety |
| `demographic_fields` | array[object] | region, age | Fields and their possible values |

---

## Quality Requirements

### Response Diversity

The generator MUST produce responses that include:

| Category | Min % | Description |
|----------|-------|-------------|
| Clear agreement | 30% | Unambiguous support for the proposal |
| Clear disagreement | 20% | Unambiguous opposition |
| Nuanced/conditional | 20% | "I agree but...", "Only if..." |
| Off-topic/tangential | 5% | Responses that miss the question |
| Low quality | 5% | Very short, vague, or unclear |
| Multi-theme | 30% | Responses touching 2+ themes |

### Response Length Distribution

| Length | Word Count | Target % |
|--------|------------|----------|
| Short | 5-50 words | 20% |
| Medium | 51-250 words | 60% |
| Long | 251-1000 words | 20% |

### Noise Injection (by level)

| Noise Type | Low | Medium | High |
|------------|-----|--------|------|
| Typos/spelling errors | 2% | 5% | 15% |
| Grammar issues | 2% | 8% | 20% |
| ALL CAPS responses | 0% | 2% | 5% |
| Emotional/heated language | 5% | 15% | 30% |
| Sarcasm/irony | 0% | 3% | 8% |

### Theme Coverage

- Each theme MUST have at least `n_responses * 0.02` mappings (2% minimum coverage)
- No single theme should exceed 40% of mappings
- Theme `X` (None of Above) should be 2-5% of responses
- Theme `Y` (No Reason) should be 1-3% of responses

---

## Validation Rules

### Referential Integrity

1. Every `response_id` in outputs MUST exist in inputs
2. Every `topic_id` in mapping MUST exist in themes
3. `respondents.jsonl` MUST have entry for every unique `response_id` across all questions

### Consistency

1. Same `response_id` MUST have same `response` text if duplicated
2. Sentiment `position` should align with mapping `stances` (AGREE → mostly POSITIVE stances)
3. `evidence_rich: YES` responses should typically be medium/long length

### Completeness

1. Every response MUST have exactly one sentiment entry
2. Every response MUST have exactly one mapping entry
3. Every response MUST have exactly one detail_detection entry

---

## Example Configuration

```yaml
# config.yaml
dataset_name: "transport_M"
topic: "public transport improvements"
size: "M"  # 1000 responses
n_questions: 3

questions:
  - text: "What improvements would you like to see to local bus services?"
    multi_choice: ["Support more buses", "Oppose changes"]
  - text: "How can we make public transport more accessible?"
  - text: "What role should cycling infrastructure play in transport planning?"

n_themes_per_question: 12
noise_level: "medium"

position_distribution:
  agree: 0.45
  disagree: 0.35
  unclear: 0.20

demographic_fields:
  - name: "region"
    values: ["England", "Scotland", "Wales", "Northern Ireland"]
    distribution: [0.84, 0.08, 0.05, 0.03]
  - name: "transport_user"
    values: ["Daily", "Weekly", "Monthly", "Rarely", "Never"]
    distribution: [0.25, 0.30, 0.20, 0.15, 0.10]
```

---

## Output Validation Checklist

Before a generated dataset is considered valid:

- [ ] All JSON/JSONL files parse without errors
- [ ] All required fields present in every record
- [ ] Referential integrity passes (response_id, topic_id links)
- [ ] Response count matches requested size (±5%)
- [ ] Theme count within 80-120% of target
- [ ] Position distribution within ±10% of requested
- [ ] No duplicate response_id within a question part
- [ ] Special themes X and Y present in all theme sets
- [ ] Date folder uses ISO format (YYYY-MM-DD)

---

## Future Extensions

Reserved for potential additions:

- `cross_cutting_themes.json` - Themes spanning multiple questions
- `response_clusters.jsonl` - Pre-computed response similarity groups
- `quality_scores.jsonl` - Per-response quality annotations
- `generation_metadata.json` - Generator version, seed, parameters used

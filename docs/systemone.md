# SystemOne classification stages

ThemeFinder's classification stages (theme mapping and detail detection) can
run on [TypeSafe's jev SystemOne model](https://docs.typesafe.ai/) instead of
an LLM. SystemOne answers typed yes/no ("noul") questions with probabilities;
it does not generate text. The generative stages (theme generation,
condensation, refinement) still run on an LLM, and `find_themes_hybrid`
combines both with the same output shape as `find_themes`.

## How the two pipelines compare

| Stage | `find_themes` (LLM) | `find_themes_hybrid` |
|---|---|---|
| 1. Theme generation | LLM | LLM, unchanged |
| 2. Theme condensation | LLM | LLM, unchanged |
| 3. Theme refinement | LLM | LLM, unchanged |
| 4. Theme mapping | LLM prompt per 20 responses, free-text JSON output | SystemOne, one request per 20 responses |
| 5. Detail detection | a second LLM pass over every response | SystemOne, merged into the same request as stage 4 |

The LLM stages return free-text JSON that must be validated and can omit
response IDs (hence the integrity checks and retry pass in the batch
processor). SystemOne answers arrive as one probability per question, keyed
to the questions that were sent, so there is nothing to parse or repair.

## Input structure

The classification entry point takes the standard responses DataFrame plus a
theme list:

```python
from themefinder import SystemOne, classify_responses_systemone

responses_df   # columns: response_id, response
themes_df      # columns: topic_id, topic ("label: description")
               # (or topic_label + topic_description)

client = SystemOne.from_env()  # reads TYPESAFE_API_KEY
classified, unprocessable = await classify_responses_systemone(
    responses_df, client, question=question, refined_themes_df=themes_df
)
```

### What is sent to SystemOne

Each request classifies a batch of responses (default 20). Shared context is
stated once in the request `state`; every question is a compact JSON pointer
naming the response and topic it is about:

```json
{
  "state": {
    "question": "…the survey question…",
    "topics": { "A": "label: description", "B": "…" },
    "topic_match_definition": "…what counts as expressing a topic…",
    "gives_reason_definition": "…what counts as a substantive answer…",
    "evidence_rich_definition": { "evidence_rich_if": "…", "not_evidence_rich_if": "…" },
    "responses": [ { "response_id": 1001, "response": "…" } ]
  },
  "questions": {
    "r1001_theme_A": { "type": "noul", "instructions": {
        "question": "Does the response express the topic? (See topic_match_definition.)",
        "response_id": 1001, "topic_id": "A" } },
    "r1001_gives_reason":  { "…": "drives the fallback labels" },
    "r1001_evidence_rich": { "…": "detail detection" }
  }
}
```

Per response that is one noul per theme, one fallback question and one
evidence question, all answered in parallel in the same request. Stating the
topic definitions once and pointing at them (rather than repeating them in
every question) roughly halved input tokens and measurably improved mapping
F1 versus verbose per-question prompts.

### What comes back and how it is used

```json
"answers": {
  "r1001_theme_A":       { "noul": 0.987 },
  "r1001_theme_B":       { "noul": 0.041 },
  "r1001_gives_reason":  { "noul": 0.994 },
  "r1001_evidence_rich": { "noul": 0.208 }
}
```

Code turns probabilities into the pipeline's output:

- themes with probability ≥ `threshold` (default 0.5) become `labels`; when
  none qualifies, the fallback is `"Other"` (substantive answer, no matching
  theme) or `"No Reason Given"`;
- `evidence_rich` is `"YES"` when its probability ≥ `detail_threshold`
  (default 0.16).

The output DataFrame keeps `theme_probabilities` and `evidence_probability`
columns, so thresholds can be re-applied later without re-querying, and
uncertain judgements (probabilities near 0.5) can be routed for review.

## Thresholds

- **Mapping threshold (0.5).** Theme probabilities are strongly bimodal, so
  this threshold is insensitive; raising it trades recall for precision.
- **Evidence threshold (0.16).** The evidence rubric is a strict conjunction,
  so probabilities cluster low while ranking responses accurately. The
  default is the stable optimum from a threshold sweep, but the optimum
  shifts with request format and data: re-check it per consultation using
  the benchmark's `best_threshold` diagnostic.

## Running and evaluating

- Minimal walkthrough: `examples/example_systemone_notebook.ipynb`
- Standalone runner: `examples/run_systemone.py`
- Benchmark (end-to-end, stage accuracy vs reference labels, scale):
  `uv run python evals/systemone_benchmark.py --llm-model gpt-4o-mini`

Install with the optional extra: `pip install 'themefinder[systemone]'`.

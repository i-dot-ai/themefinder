# LLM-as-Judge Improvement Plan

**Notional Implementation Roadmap** - Based on research of codebase, Langfuse capabilities, synthetic data quality, and 2025 best practices.

---

## Executive Summary

### Current State

| Aspect | Status | Issue |
|--------|--------|-------|
| Judge model | GPT-4o (hard-coded in metrics.py) | Same model for task and judge = potential bias |
| Scoring | 0-5 scale | Too granular, inconsistent |
| Prompts | Basic instructions | No few-shot examples, no CoT |
| Ground truth | 100% LLM-generated | Circular evaluation (LLM vs LLM) |
| Reliability | No shuffling, no position swapping | ~40% positional bias |
| Cost tracking | Basic Langfuse integration | No semantic caching |
| Error handling | Returns 0.0 on failure | No retry differentiation |

### Target State

| Aspect | Target | Improvement |
|--------|--------|-------------|
| Judge model | Separate, configurable judge | Eliminates self-evaluation bias |
| Scoring | Binary + aggregation | 10-15pp consistency improvement |
| Prompts | Few-shot with CoT | 12pp reliability improvement |
| Ground truth | Hybrid (LLM + validation) | Reduced circularity |
| Reliability | Shuffle + position swap | 40% bias reduction |
| Cost | Hierarchical + caching | 50-60% cost reduction |
| Error handling | Tiered retry with fallback | 96%+ ECR@1 |

---

## Improvement Proposals

### P1: Shuffle Theme Order in Prompts

**Problem**: First themes in prompt get more attention, later themes trail off.

**Solution**: Randomise theme order before each evaluation.

```python
import random

def evaluate_with_shuffle(themes: list, prompt_template: str, llm) -> dict:
    shuffled = themes.copy()
    random.shuffle(shuffled)
    return llm.invoke(prompt_template.format(themes=shuffled))
```

| Dimension | Assessment |
|-----------|------------|
| **Cost (Money)** | Zero - no additional API calls |
| **Cost (Time)** | 1-2 hours implementation |
| **Complexity** | Trivial - single function change |
| **Semantic Complexity** | None |
| **Models/Patterns** | None - pure Python |
| **Impact** | High - eliminates positional bias |

**Files to modify**: `evaluators.py`, `utils.py`

---

### P2: Binary Judgments with Aggregation

**Problem**: 0-5 scale has unclear distinctions (is this a 3 or 4?). Research shows binary is more reliable.

**Solution**: Replace 0-5 with binary per-theme decisions, then aggregate.

**Current prompt** (generation_eval.txt):
```
Assign a score from 0 to 5 based on the following criteria:
5: Perfect match
4: Strong match
...
```

**Proposed prompt**:
```
For each theme in LIST 1, answer these binary questions:

1. Does this theme have ANY match in LIST 2?
   Answer: YES / NO

2. If YES, how strong is the match?
   Answer: STRONG (same concept) / PARTIAL (related but different scope)

Output JSON:
{
  "theme_label": {"has_match": true, "strength": "STRONG", "matched_to": "other_label"},
  ...
}
```

**Aggregation logic**:
```python
def aggregate_binary_scores(results: list[dict]) -> float:
    """Convert binary decisions to 0-5 scale for backwards compatibility."""
    scores = []
    for theme, judgment in results.items():
        if not judgment["has_match"]:
            scores.append(0)
        elif judgment["strength"] == "STRONG":
            scores.append(5)
        else:  # PARTIAL
            scores.append(3)
    return np.mean(scores)
```

| Dimension | Assessment |
|-----------|------------|
| **Cost (Money)** | Zero - same number of API calls |
| **Cost (Time)** | 4-6 hours (prompt rewrite + aggregation logic) |
| **Complexity** | Low - prompt engineering + Python |
| **Semantic Complexity** | Medium - need to validate binary prompts produce consistent results |
| **Models/Patterns** | Prompt engineering, structured output |
| **Impact** | High - 10-15pp consistency improvement per research |

**Files to modify**: `evals/prompts/generation_eval.txt`, `evaluators.py`

---

### P3: Add Chain-of-Thought Reasoning

**Problem**: No visibility into why judge made decisions. Users complain: *"I'm left wondering why the themes are chosen"*

**Solution**: Require reasoning before score. Makes judgments auditable and improves reliability by 10-15%.

**Updated prompt structure**:
```
For theme "{theme_label}":

Step 1: Identify the core concept
- What is this theme fundamentally about?

Step 2: Search for matches
- Which theme(s) in the comparison list address the same concept?

Step 3: Assess alignment
- What aligns between them?
- What differs?

Step 4: Judgment
- Decision: MATCH / NO_MATCH
- If MATCH, strength: STRONG / PARTIAL
- Reasoning summary: <1-2 sentences>

Output JSON:
{
  "theme_label": {
    "reasoning": "This theme about X matches theme Y because...",
    "decision": "MATCH",
    "strength": "STRONG"
  }
}
```

| Dimension | Assessment |
|-----------|------------|
| **Cost (Money)** | +20-40% tokens per evaluation (reasoning text) |
| **Cost (Time)** | 4-6 hours implementation |
| **Complexity** | Low - prompt modification |
| **Semantic Complexity** | Low - CoT is well-understood pattern |
| **Models/Patterns** | Chain-of-thought prompting |
| **Impact** | Medium-High - 10-15% reliability improvement, enables debugging |

**Files to modify**: `evals/prompts/*.txt`

**Note**: Can be combined with P2 for single implementation.

---

### P4: Separate Judge Model Configuration

**Problem**: Current implementation uses same LLM for task execution and judging. This creates:
1. Self-evaluation bias (5-7% score inflation)
2. No flexibility to use cheaper judge for high-volume evals
3. Hard-coded GPT-4o in `metrics.py`

**Solution**: Make judge model configurable, separate from task model.

**Implementation**:

```python
# benchmark.py - Add judge_model to BenchmarkRunner
@dataclass
class BenchmarkConfig:
    models: list[ModelConfig]
    judge_model: ModelConfig | None = None  # If None, uses GPT-4o default
    datasets: list[str]
    eval_types: list[str]
    runs_per_model: int = 3

# evaluators.py - Accept judge as parameter (already does this)
def create_groundedness_evaluator(judge_llm: BaseLLM):
    """Judge LLM is now explicitly passed, not same as task LLM."""
    ...

# CLI addition
parser.add_argument(
    "--judge-model",
    default="gpt-4o-mini",
    help="Model to use for LLM-as-judge evaluations"
)
```

**Recommended judge models** (by use case):

| Use Case | Judge Model | Cost/1K evals | Accuracy |
|----------|-------------|---------------|----------|
| Development | GPT-4o Mini | $1.01 | 96.6% ECR@1 |
| Production | GPT-4o | $3-5 | Highest |
| High-volume | Qwen-2.5-72B | $0.45-0.63 | 85%+ |
| Budget | Prometheus-2 7B | $0.01-0.05 | 80%+ |

| Dimension | Assessment |
|-----------|------------|
| **Cost (Money)** | Potentially -50% to -90% depending on judge choice |
| **Cost (Time)** | 6-8 hours implementation |
| **Complexity** | Medium - config changes, CLI updates |
| **Semantic Complexity** | Low |
| **Models/Patterns** | Dependency injection, factory pattern |
| **Impact** | High - eliminates bias, enables cost optimization |

**Files to modify**: `benchmark.py`, `evaluators.py`, `metrics.py`, CLI argument parsing

---

### P5: Few-Shot Calibration Examples

**Problem**: Judges have no examples of "good" vs "bad" theme matches. Research shows few-shot improves consistency from 65% → 77.5% (12.5pp).

**Solution**: Add calibrated examples to judge prompts.

**Example calibration set**:
```
## Calibration Examples

Example 1 - STRONG MATCH (Score: 5)
Theme A: "Financial concerns about housing costs"
Theme B: "Affordability worries for first-time buyers"
Reasoning: Both address financial barriers to housing. Different framing but same core concern.

Example 2 - PARTIAL MATCH (Score: 3)
Theme A: "Infrastructure investment needs"
Theme B: "Road maintenance concerns"
Reasoning: Roads are infrastructure, but Theme A is broader. Partial overlap only.

Example 3 - NO MATCH (Score: 0)
Theme A: "Environmental impact of development"
Theme B: "Community consultation process"
Reasoning: Completely different concerns - one about ecology, one about governance.

Example 4 - TRICKY CASE (Score: 2)
Theme A: "Support for vulnerable populations"
Theme B: "Elderly care provisions"
Reasoning: Elderly are one vulnerable group, but Theme A could include many others. Weak match.
```

**Source for examples**:
1. Hand-craft from DWP PIP theme merging document
2. Extract from survivorship data (which themes users merged)
3. Generate with GPT-4, validate manually

| Dimension | Assessment |
|-----------|------------|
| **Cost (Money)** | +10-20% tokens per evaluation (example text in prompt) |
| **Cost (Time)** | 8-12 hours (example creation + validation) |
| **Complexity** | Medium - requires domain expertise for examples |
| **Semantic Complexity** | High - examples must be representative, not misleading |
| **Models/Patterns** | Few-shot prompting |
| **Impact** | High - 12pp consistency improvement |

**Files to modify**: `evals/prompts/*.txt`

**Dependency**: Need access to real consultation data or survivorship database for quality examples.

---

### P6: Title Specificity Metric

**Problem**: Users complain titles are "very catch-all" and "non-specific". Not currently measured.

**Solution**: Add standalone evaluator for title quality.

**New prompt** (`title_specificity_eval.txt`):
```
Evaluate theme title specificity. A good title:
- Conveys the core concern without reading the description
- Is specific enough to distinguish from other themes
- Would help an analyst immediately understand what responses this covers

For each theme, answer:
1. Could this title apply to many different themes? (YES = vague)
2. Does it mention specific concepts/policies/impacts?
3. Would two analysts agree on what this covers?

Examples:
- VAGUE: "Concerns about the process"
- SPECIFIC: "Delays in planning permission approvals"

- VAGUE: "Financial issues"
- SPECIFIC: "Stamp duty burden for first-time buyers"

Output JSON:
{
  "theme_label": {
    "specificity": "SPECIFIC" | "VAGUE",
    "reasoning": "..."
  }
}
```

**Metric aggregation**:
```python
def title_specificity_score(results: dict) -> float:
    """Return fraction of titles rated as SPECIFIC."""
    specific_count = sum(1 for r in results.values() if r["specificity"] == "SPECIFIC")
    return specific_count / len(results)
```

| Dimension | Assessment |
|-----------|------------|
| **Cost (Money)** | +1 API call per theme set (~$0.01-0.02) |
| **Cost (Time)** | 4-6 hours |
| **Complexity** | Low - new evaluator following existing pattern |
| **Semantic Complexity** | Medium - "specific" is somewhat subjective |
| **Models/Patterns** | LLM-as-judge, categorical output |
| **Impact** | Medium - addresses key user pain point |

**Files to create**: `evals/prompts/title_specificity_eval.txt`
**Files to modify**: `evaluators.py`, `eval_generation.py`

---

### P7: Semantic Redundancy Detection

**Problem**: "30 themes feels overwhelming, especially when many are essentially the same". Not currently measured.

**Solution**: Use embeddings to detect semantic overlap between themes in same set.

**Implementation**:
```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def calculate_redundancy_score(themes: list[dict], threshold: float = 0.85) -> dict:
    """Detect semantically similar themes within a single theme set."""
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Embed theme descriptions
    texts = [f"{t['label']}: {t['description']}" for t in themes]
    embeddings = model.encode(texts)

    # Calculate pairwise similarity
    similarity_matrix = cosine_similarity(embeddings)

    # Find redundant pairs (above threshold, excluding self)
    redundant_pairs = []
    for i in range(len(themes)):
        for j in range(i + 1, len(themes)):
            if similarity_matrix[i, j] > threshold:
                redundant_pairs.append({
                    "theme_a": themes[i]["label"],
                    "theme_b": themes[j]["label"],
                    "similarity": float(similarity_matrix[i, j])
                })

    return {
        "redundancy_score": 1 - (len(redundant_pairs) / max(len(themes), 1)),
        "redundant_pairs": redundant_pairs,
        "n_themes": len(themes),
        "n_redundant": len(redundant_pairs)
    }
```

| Dimension | Assessment |
|-----------|------------|
| **Cost (Money)** | ~$0 (local embedding model) or $0.0001 via API |
| **Cost (Time)** | 4-6 hours |
| **Complexity** | Medium - requires embedding infrastructure |
| **Semantic Complexity** | Medium - threshold tuning needed |
| **Models/Patterns** | Sentence embeddings, cosine similarity |
| **Impact** | High - addresses key user pain point, enables monitoring |

**Dependencies**: `sentence-transformers` package
**Files to modify**: `evaluators.py`, `eval_generation.py`, `eval_condensation.py`

---

### P8: Variance/Consistency Metrics

**Problem**: "Sometimes you put it in and you got 10 and sometimes you got 60" themes. No consistency measurement.

**Solution**: Run N times, measure variance in theme count and semantic overlap.

**Implementation**:
```python
@dataclass
class ConsistencyMetrics:
    theme_count_mean: float
    theme_count_std: float
    theme_count_cv: float  # Coefficient of variation
    jaccard_similarity_mean: float  # Between runs
    semantic_overlap_mean: float

def measure_consistency(
    task_fn: Callable,
    input_data: dict,
    n_runs: int = 10
) -> ConsistencyMetrics:
    """Run task N times, measure output consistency."""
    results = []
    for _ in range(n_runs):
        output = task_fn(input_data)
        results.append(output["themes"])

    # Theme count variance
    counts = [len(r) for r in results]

    # Jaccard similarity between all pairs
    jaccard_scores = []
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            labels_i = set(t["label"] for t in results[i])
            labels_j = set(t["label"] for t in results[j])
            jaccard = len(labels_i & labels_j) / len(labels_i | labels_j)
            jaccard_scores.append(jaccard)

    return ConsistencyMetrics(
        theme_count_mean=np.mean(counts),
        theme_count_std=np.std(counts),
        theme_count_cv=np.std(counts) / np.mean(counts),
        jaccard_similarity_mean=np.mean(jaccard_scores),
        semantic_overlap_mean=calculate_semantic_overlap(results)
    )
```

| Dimension | Assessment |
|-----------|------------|
| **Cost (Money)** | N × base cost (e.g., 10× for N=10) |
| **Cost (Time)** | 6-8 hours implementation |
| **Complexity** | Medium - new metric type, changes to benchmark runner |
| **Semantic Complexity** | Medium - need to define "acceptable" variance |
| **Models/Patterns** | Statistical analysis, Jaccard similarity |
| **Impact** | High - addresses core reliability concern |

**Files to modify**: `benchmark.py`, new `consistency_metrics.py`

**Note**: Expensive to run (N× cost), recommend as separate "consistency mode" not default.

---

### P9: Hierarchical Evaluation (Cheap Filter → Expensive Judge)

**Problem**: Every evaluation uses expensive GPT-4o, even for clear pass/fail cases.

**Solution**: Multi-stage evaluation with semantic filtering.

**Architecture**:
```
Theme Set → Embedding Check → Cache Hit?
                         ├─ Yes → Return cached score (FREE)
                         └─ No → Stage 2
                                    ↓
                    Heuristic Check (redundancy, count, etc.)
                         ├─ Clear Pass/Fail → Return ($0.0001)
                         └─ Borderline → Stage 3
                                    ↓
                         LLM Judge → Score ($0.01-0.02)
```

**Implementation**:
```python
class HierarchicalEvaluator:
    def __init__(
        self,
        cache: SemanticCache,
        heuristic_fn: Callable,
        llm_judge: BaseLLM,
        cache_threshold: float = 0.92,
        heuristic_confidence: float = 0.8
    ):
        self.cache = cache
        self.heuristic_fn = heuristic_fn
        self.llm_judge = llm_judge

    def evaluate(self, themes: list[dict], expected: list[dict]) -> EvalResult:
        # Stage 1: Cache check
        cache_key = self._make_cache_key(themes, expected)
        cached = self.cache.get(cache_key, threshold=self.cache_threshold)
        if cached:
            return EvalResult(score=cached, source="cache", cost=0)

        # Stage 2: Heuristic check
        heuristic_result = self.heuristic_fn(themes, expected)
        if heuristic_result.confidence > self.heuristic_confidence:
            self.cache.set(cache_key, heuristic_result.score)
            return EvalResult(
                score=heuristic_result.score,
                source="heuristic",
                cost=0.0001
            )

        # Stage 3: LLM judge
        llm_result = self._run_llm_judge(themes, expected)
        self.cache.set(cache_key, llm_result.score)
        return EvalResult(
            score=llm_result.score,
            source="llm",
            cost=0.02,
            reasoning=llm_result.reasoning
        )
```

**Cost analysis** (1000 evaluations):
- 30% cache hits: 300 × $0 = $0
- 40% heuristic pass: 400 × $0.0001 = $0.04
- 30% LLM judge: 300 × $0.02 = $6.00
- **Total: $6.04 vs $20.00 direct = 70% savings**

| Dimension | Assessment |
|-----------|------------|
| **Cost (Money)** | -50% to -70% per evaluation |
| **Cost (Time)** | 16-24 hours (significant refactor) |
| **Complexity** | High - new architecture, caching infrastructure |
| **Semantic Complexity** | High - cache key design, threshold tuning |
| **Models/Patterns** | Semantic caching, tiered evaluation, embeddings |
| **Impact** | Very High - major cost reduction at scale |

**Dependencies**:
- Semantic cache (GPTCache or custom with Qdrant)
- Embedding model (sentence-transformers or OpenAI)

**Files to create**: `hierarchical_evaluator.py`, `semantic_cache.py`
**Files to modify**: `benchmark.py`, `evaluators.py`

---

### P10: Langfuse Native LLM-as-Judge Integration

**Problem**: Custom judge implementation doesn't leverage Langfuse's built-in capabilities (templates, batching, cost tracking).

**Solution**: Migrate to Langfuse's evaluator system where appropriate.

**Langfuse capabilities discovered**:
1. 8 pre-built templates (hallucination, relevance, etc.)
2. Custom evaluator creation via UI
3. Execution tracing (October 2025) - full visibility into judge calls
4. Sampling (evaluate 5% of traces to reduce cost)
5. Batch evaluation on existing data (February 2025)

**Implementation**:
```python
from langfuse import Langfuse

def setup_langfuse_evaluators(client: Langfuse):
    """Configure Langfuse native evaluators for theme evaluation."""

    # Use Langfuse's built-in evaluator for relevance
    # Configure via Langfuse UI with custom prompt

    # For custom theme evaluators, create via API when supported
    # Currently UI-only for custom evaluators

    # Attach to traces
    with client.trace(name="theme_generation") as trace:
        # ... run task ...

        # Langfuse auto-runs configured evaluators
        # Results visible in Langfuse dashboard
        pass
```

**Migration approach**:
1. Keep existing evaluators for complex theme logic
2. Use Langfuse native for standard metrics (hallucination, coherence)
3. Leverage Langfuse sampling for cost control in production

| Dimension | Assessment |
|-----------|------------|
| **Cost (Money)** | -20% to -50% via sampling |
| **Cost (Time)** | 8-12 hours |
| **Complexity** | Medium - integration work, UI configuration |
| **Semantic Complexity** | Low - Langfuse handles complexity |
| **Models/Patterns** | Langfuse SDK, evaluator configuration |
| **Impact** | Medium - better observability, easier cost control |

**Files to modify**: `langfuse_utils.py`, `benchmark.py`

---

### P11: Ground Truth Validation Layer

**Problem**: Ground truth is 100% LLM-generated, creating circular evaluation (LLM vs LLM). No human validation.

**Solution**: Add validation layer for synthetic data quality.

**Approaches**:

**A. Intruder Detection Test** (automated):
```python
def intruder_test(theme: dict, responses: list[str], n_intruders: int = 1) -> float:
    """Insert responses from other themes, ask LLM to identify intruders."""
    # Get responses assigned to this theme
    correct_responses = responses[:4]

    # Get intruder from different theme
    intruder = get_response_from_other_theme(theme)

    # Shuffle and present
    test_set = correct_responses + [intruder]
    random.shuffle(test_set)

    # Ask LLM to identify which doesn't belong
    result = llm.invoke(f"""
    Theme: {theme['label']} - {theme['description']}

    Which of these responses does NOT belong to this theme?
    {format_responses(test_set)}

    Answer with the response number only.
    """)

    # Check if intruder was correctly identified
    return 1.0 if result == intruder_position else 0.0
```

**B. Human Spot-Check** (manual):
- Sample 5% of ground truth mappings
- Have analyst verify response-theme assignments
- Track agreement rate over time

**C. Cross-Validation with Real Data**:
- Compare synthetic themes against real consultation themes
- Measure semantic similarity
- Flag synthetic themes with no real-world equivalent

| Dimension | Assessment |
|-----------|------------|
| **Cost (Money)** | A: +10% eval cost, B: analyst time, C: depends on data access |
| **Cost (Time)** | 12-20 hours |
| **Complexity** | Medium-High |
| **Semantic Complexity** | High - defining "valid" ground truth |
| **Models/Patterns** | Intruder detection, cross-validation |
| **Impact** | Medium - reduces circularity concern |

**Dependencies**: Access to real consultation data for option C.

---

## Implementation Phases

### Phase 1: Quick Wins (Week 1)

| Proposal | Effort | Impact | Cost Change |
|----------|--------|--------|-------------|
| P1: Shuffle theme order | 2h | High | $0 |
| P2: Binary judgments | 6h | High | $0 |
| P3: Chain-of-thought | 4h | Medium | +20% tokens |

**Total effort**: ~12 hours
**Expected outcome**: 20-25pp consistency improvement

### Phase 2: Measurement (Week 2)

| Proposal | Effort | Impact | Cost Change |
|----------|--------|--------|-------------|
| P6: Title specificity | 5h | Medium | +$0.02/set |
| P7: Semantic redundancy | 5h | High | ~$0 |
| P4: Separate judge model | 8h | High | Variable |

**Total effort**: ~18 hours
**Expected outcome**: New metrics for user pain points, cost flexibility

### Phase 3: Reliability (Week 3-4)

| Proposal | Effort | Impact | Cost Change |
|----------|--------|--------|-------------|
| P5: Few-shot calibration | 12h | High | +15% tokens |
| P8: Consistency metrics | 8h | High | N× for consistency runs |
| P10: Langfuse native | 10h | Medium | -20% via sampling |

**Total effort**: ~30 hours
**Expected outcome**: Production-ready reliability, observability

### Phase 4: Scale Optimization (Month 2+)

| Proposal | Effort | Impact | Cost Change |
|----------|--------|--------|-------------|
| P9: Hierarchical evaluation | 24h | Very High | -50% to -70% |
| P11: Ground truth validation | 16h | Medium | +10% |

**Total effort**: ~40 hours
**Expected outcome**: Cost-efficient at scale, validated ground truth

---

## Cost Summary

### Current State
- ~$0.02-0.03 per theme set evaluation
- No caching, no sampling
- Same model for task and judge

### Phase 1-2 Target
- ~$0.015-0.025 per evaluation
- Configurable judge model
- New quality metrics

### Phase 3-4 Target
- ~$0.005-0.015 per evaluation (50-70% reduction)
- Hierarchical evaluation
- Semantic caching
- Langfuse sampling in production

### Cost Comparison Table

| Approach | Cost/1K Evals | Accuracy | Use Case |
|----------|---------------|----------|----------|
| Current (GPT-4o) | $20-30 | High | Development |
| GPT-4o Mini judge | $10-15 | High | Production |
| Hierarchical + cache | $5-10 | High | Scale |
| Prometheus-2 7B | $1-5 | Good | High volume |

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Binary judgments lose nuance | Medium | Medium | Keep 0-5 as fallback, A/B test |
| Few-shot examples bias judge | Low | High | Diverse examples, validation |
| Semantic cache false positives | Medium | Medium | Conservative threshold (0.92+) |
| Prometheus-2 quality drop | Medium | Medium | Benchmark before switching |
| Ground truth circularity | High | Medium | Intruder tests, human spot-checks |

---

## Decision Points

### Requires User Input

1. **Judge model selection**: GPT-4o Mini (recommended) vs Qwen vs Prometheus?
2. **Consistency runs**: How many N for variance measurement? (Recommend N=10)
3. **Few-shot examples**: Need real consultation data or synthesise from DWP PIP doc?
4. **Ground truth validation**: Intruder test only, or human spot-checks too?

### Technical Decisions

1. **Caching backend**: GPTCache vs custom Qdrant integration?
2. **Embedding model**: sentence-transformers (local) vs OpenAI (API)?
3. **Langfuse evaluator strategy**: Custom only vs hybrid with native?

---

## Next Steps

1. **Immediate**: Implement P1 (shuffle) and P2 (binary) - low effort, high impact
2. **This week**: Add P6 (title specificity) and P7 (redundancy) metrics
3. **Next week**: Implement P4 (separate judge) and P5 (few-shot)
4. **Month 2**: Build hierarchical evaluation (P9) if scale justifies

---

## Appendix: File Modification Summary

| File | Proposals Affecting |
|------|---------------------|
| `evaluators.py` | P1, P2, P3, P4, P6, P7 |
| `benchmark.py` | P4, P8, P9, P10 |
| `evals/prompts/generation_eval.txt` | P2, P3, P5 |
| `evals/prompts/condensation_eval.txt` | P3, P5 |
| `evals/prompts/refinement_eval.txt` | P3, P5 |
| `langfuse_utils.py` | P10 |
| `metrics.py` | P4 |
| `utils.py` | P1 |
| **New files** | |
| `evals/prompts/title_specificity_eval.txt` | P6 |
| `consistency_metrics.py` | P8 |
| `hierarchical_evaluator.py` | P9 |
| `semantic_cache.py` | P9 |
| `ground_truth_validator.py` | P11 |

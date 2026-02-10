# LLM-as-Judge: Analysis & Improvement Routes

**Meeting with Mitch & Katie** - incorporating feedback from Mitch call (3 Feb) and Katie's user research reports

---

## User Research Insights (Katie's Reports)

### DESNZ Pilot 1 - Theme Sign Off Experience (Sep 2025)

**Context**: Small consultation (<100 responses), 8-person team, 12-week timeline pressure

> **Caveat**: This was a small consultation where the team had already manually read all responses before using Consult. Their pain points reflect having pre-existing mental models of the themes. For large consultations (10,000-20,000+ responses), manual reading isn't feasible - the value proposition and user needs are fundamentally different. Take these insights as directional, not definitive.

#### What Users Value

| Finding | Implication for Eval |
|---------|---------------------|
| **Theme descriptions more useful than titles** | Evaluate description quality separately from title quality |
| **Themes were "broadly as expected"** | Ground truth alignment is working |
| **Team sessions helped align approaches** | Inter-rater agreement matters |

#### User Pain Points (Direct Quotes)

> "I may as well have done it manually" - on themes grouping agree/disagree together

> "Theme titles are very catch-all and non-specific"

> "I'm left wondering why the themes are chosen, what is it trying to achieve"

> "I want it to tell me 'majority of people said this or minority had that view'"

#### Key Problems to Measure

| Problem | Current Measurement | Gap |
|---------|---------------------|-----|
| **Sentiment not separated** | Not measured | Need sentiment-aware theme evaluation |
| **Too many themes (30+)** | Theme count variance | Need target range metric |
| **Themes "essentially the same"** | Not measured | Need semantic redundancy metric |
| **Titles too vague** | Partially (clarity) | Need specificity metric |
| **Lack of AI transparency** | Not measured | Need explainability in judge reasoning |
| **Risk-averse deletion** | Not measured | Track survivorship patterns |

### DWP PIP Theme Mapping Lock-In

**Context**: Large consultation (~14,500 responses), ~3,500 raw themes → 30 final themes

#### Theme Merging Examples

The document shows concrete examples of theme quality issues:

**Similar themes that should merge** (emotional focus):
- "Humiliating current system"
- "Less stressful assessments"
- "Inhumane current assessments"

→ Merged to: "Dignified Assessments: A More Humane Approach"

**Similar themes that should merge** (process focus):
- "Speeding up the process"
- "Simplifying the process"

→ Merged to: "Streamlining Assessments for Efficiency and Fairness"

**Key insight**: Granularity is contextual. Same themes could be merged or kept separate "depending on the desired level of theme granularity."

#### Evaluation Implications

1. **Redundancy detection** must account for granularity preferences
2. **Theme naming** matters - "Improvement over current system" vs "Dignified Assessments"
3. **30 themes still has duplication** - condensation not aggressive enough

---

## Key Insights from Mitch (3 Feb 2026)

### 1. Make Judge Tasks More Atomic

> "The closer your judgment task is to the original task, the harder it is for you to have any faith that the judge can more meaningfully do the task."

**Current approach**: Pass two lists, ask for multiple 1-5 judgments in one call

**Recommended approach**:
- One call per theme (not per list)
- Binary choices rather than 1-5 scales
- Aggregate binary decisions to get final scores

**Trade-off**: Cost vs reliability. Semantic clustering could reduce calls while maintaining quality.

### 2. Positional Bias - Shuffle Theme Order

> "Even if you don't run it for every theme, you could shuffle the order... the distributions should reflect underlying performance, not order."

**Action**: Randomise theme order in prompts before evaluation. This is low-cost and high-impact.

### 3. Measure Variance, Not Just Accuracy

> "Sometimes you put it in and you got 10 [themes] and sometimes you got 60... in those 60 there was a lot of repetition."

**Key metrics to add**:
- Theme count variance across runs (target: 10-20, not 10-70)
- Semantic duplication within theme sets
- Consistency across multiple runs of same input

This is closer to **monitoring** than evaluation - measuring product reliability.

### 4. Theme Set Quality in Isolation

Not just "do generated themes match ground truth" but standalone quality metrics:
- Is there internal overlap/redundancy?
- Are themes at appropriate granularity?
- Would an analyst find these useful?

### 5. Multiple Runs + Aggregation

Michael's idea for mapping: run twice, take union of assigned themes. There's likely an optimal number of runs before over-assignment degrades quality.

**Experiment**: Run N times, aggregate, measure performance curve.

### 6. Survivorship Data from Consult

Users indicate which themes they:
- Kept unchanged
- Combined with others
- Split into multiple themes
- Deleted entirely

**Action**: Get read-only database access to analyse editorial patterns. Even simple heuristics could improve prompts.

---

## Quality Dimensions (Combined Sources)

### From Mitch's Evaluation Framework

| Dimension | Description |
|-----------|-------------|
| **Clarity** | Are the theme name and description understandable? |
| **Coherence** | Do sampled responses for this theme clearly belong together? |
| **Actionability** | Would this help an analyst/policy team act? |
| **Non-overlap** | Is the theme clearly distinct from other themes in the same list? |

### From User Research (Katie)

| Dimension | User Language | How to Measure |
|-----------|---------------|----------------|
| **Specificity** | "very catch-all", "non-specific" | Title concreteness score |
| **Sentiment separation** | "agree and disagree grouped together" | Check if themes split by position |
| **Transparency** | "why the themes are chosen" | Chain-of-thought reasoning |
| **Quantifiability** | "majority said this, minority that" | Response count per theme |
| **Appropriate count** | "30 themes feels overwhelming" | Theme count vs consultation size |

### Proposed Unified Rubric

```
For each theme, evaluate:

1. CLARITY (1-5)
   - Is the title specific and descriptive? (not "catch-all")
   - Is the description accurate and useful?

2. COHERENCE (1-5)
   - Do example responses clearly belong together?
   - Would inserting an unrelated response be obvious?

3. DISTINCTIVENESS (1-5)
   - Is this theme clearly different from others in the list?
   - Could it be merged with another theme without information loss?

4. ACTIONABILITY (1-5)
   - Could a policy team act on this theme?
   - Does it represent a clear viewpoint or concern?

5. SENTIMENT HANDLING (Binary)
   - If responses have mixed sentiment, is the theme appropriately split?
   - Or is mixed sentiment handled transparently in the description?
```

---

## Current Implementation Summary

### Architecture
- **Single judge**: GPT-4o with `temperature=0.0`
- **Scoring**: 0-5 absolute scale (too granular per Mitch)
- **No chain-of-thought**: Outputs scores without reasoning
- **No shuffling**: Positional bias affects results

### Evaluators in Use

| Evaluator | Metrics | Type |
|-----------|---------|------|
| `generation_eval.txt` | Groundedness, Coverage | LLM judge (bidirectional) |
| `condensation_eval.txt` | Compression Quality, Information Retention, Rare Topic Preservation | LLM judge |
| `refinement_eval.txt` | Information Retention, Response References, Distinctiveness, Fluency | LLM judge |
| `sentiment_accuracy` | Accuracy | Deterministic |
| `mapping_f1` | F1 Score | Deterministic |

---

## Research Findings (2025)

### Known Biases

| Issue | Impact |
|-------|--------|
| Position bias | ~40% inconsistency in pairwise comparisons |
| Verbosity bias | ~15% score inflation for longer outputs |
| Self-enhancement bias | 5-7% boost when models evaluate their own family |
| Chain-of-thought | 10-15% reliability improvement when added |

### Key Insights

- **Simpler scales work better**: Binary or 0-1-2 more stable than 0-5
- **Absolute scoring more robust** than pairwise (9% flip rate vs 35%)
- **Human-LLM agreement**: GPT-4 achieves ~85% (higher than human-human at 81%)

---

## Proposed Improvements

### P1: Immediate Actions

#### 1. Shuffle Theme Order Before Evaluation
Zero-cost reliability improvement. Randomise theme order in all judge prompts.

```python
import random

def evaluate_with_shuffle(themes, judge_prompt, llm):
    shuffled = themes.copy()
    random.shuffle(shuffled)
    return llm.invoke(judge_prompt.format(themes=shuffled))
```

#### 2. Simplify to Binary Judgments
Replace 0-5 with binary per-theme decisions, then aggregate.

```
For theme X: Does it match any theme in the comparison list?
Answer: YES / NO

If YES, which theme? [select from list]
How strong is the match? STRONG / PARTIAL
```

#### 3. Add Variance Metrics
Track across multiple runs:
- Theme count (mean, std)
- Semantic similarity matrix between runs
- Jaccard similarity of theme sets

#### 4. Add Chain-of-Thought
Require reasoning before score. Makes judgments auditable. Addresses user pain point: *"I'm left wondering why the themes are chosen"*

```
Before scoring, explain:
1. Core concept of this theme
2. Closest match in comparison list
3. What aligns / what differs
4. Your judgment

Reasoning: <analysis>
Decision: MATCH / NO_MATCH
```

#### 5. Add Title Specificity Metric
Address user complaint about "catch-all" and "non-specific" titles.

```
Evaluate theme title:
- Is it specific enough to distinguish from other themes?
- Does it convey the core concern without reading the description?
- Would an analyst immediately understand what responses this covers?

Score: SPECIFIC / VAGUE
```

### P2: Medium-Term

#### 6. Standalone Theme Set Quality
Evaluate theme lists without ground truth comparison:

```
Given this theme list, evaluate:
1. Internal redundancy: Do any themes overlap significantly?
2. Granularity balance: Are themes at similar levels of specificity?
3. Coverage gaps: Are there obvious missing themes?
4. Actionability: Could a policy team act on these?
5. Sentiment handling: Are agree/disagree positions appropriately separated?
```

#### 7. Intruder Detection (Automated)
LLM-based coherence testing:

```
Here are 5 responses assigned to Theme X:
[response 1]
[response 2]
[response 3 - INTRUDER from different theme]
[response 4]
[response 5]

Which response doesn't belong? Why?
```

#### 8. Theme Count Appropriateness
Based on user feedback about "30 themes feels overwhelming":

```
Given:
- Consultation size: N responses
- Question complexity: [simple yes/no | open-ended | multi-part]

Expected theme count range: X-Y

Actual theme count: Z

Score: APPROPRIATE / TOO_MANY / TOO_FEW
```

#### 9. Multi-Run Aggregation Experiment
Test Michael's hypothesis:
- Run theme generation N times (N=1,2,3,5,10)
- For mapping: take union of assignments
- Measure precision/recall curve
- Find optimal N

### P3: Longer-Term

#### 10. Survivorship Analysis
Analyse Consult database for editorial patterns:
- Which AI themes survive unchanged?
- What predicts combination/splitting?
- Can we learn heuristics for better initial themes?

#### 11. Multi-Judge Ensemble
Use 2-3 different models, aggregate via majority vote:
- GPT-4o
- Claude Sonnet
- Gemini Pro

Trade-off: 3x cost, but reduces single-model bias.

#### 12. Sentiment-Aware Evaluation
Address the key user pain point: themes grouping agree/disagree together.

```
For themes about a policy proposal:
- Are supporting arguments in separate themes from opposing arguments?
- Or is mixed sentiment clearly explained in description?
- Does the theme structure enable "majority said X, minority said Y" reporting?
```

---

## Missing Evaluation Dimensions

| Dimension | Description | User Evidence | Priority |
|-----------|-------------|---------------|----------|
| **Consistency** | Same input → similar outputs | "10 vs 60 themes" (Mitch) | **HIGH** |
| **Semantic redundancy** | Internal overlap | "essentially the same" (DESNZ) | **HIGH** |
| **Title specificity** | Concrete vs vague | "catch-all", "non-specific" (DESNZ) | **HIGH** |
| **Sentiment separation** | Agree/disagree split | "may as well have done it manually" (DESNZ) | **HIGH** |
| **Transparency** | Explainable decisions | "why themes are chosen" (DESNZ) | Medium |
| **Theme count** | Appropriate for consultation size | "30 themes overwhelming" (DESNZ) | Medium |
| **Actionability** | Policy team can act | Mitch framework | Medium |
| **Quantifiability** | Supports "majority/minority" reporting | "want quantitative info" (DESNZ) | Low |

---

## Implementation Priority

| Improvement | Impact | Effort | Source |
|-------------|--------|--------|--------|
| Shuffle theme order | High | Trivial | Mitch |
| Binary judgments | High | Low | Mitch + research |
| Title specificity metric | High | Low | Katie (DESNZ) |
| Variance metrics | High | Medium | Mitch |
| Chain-of-thought | Medium | Low | Research + Katie |
| Standalone quality metrics | High | Medium | Mitch |
| Sentiment separation check | High | Medium | Katie (DESNZ) |
| Intruder detection | Medium | Medium | Mitch |
| Theme count appropriateness | Medium | Low | Katie (DESNZ) |
| Multi-run aggregation | Medium | Medium | Michael via Mitch |
| Survivorship analysis | High | Medium | Mitch |

---

## Next Steps

1. **This week**: Implement shuffle + binary judgments in eval pipeline
2. **This week**: Add variance metrics (N=30 runs on small dataset)
3. **This week**: Add title specificity and sentiment separation checks
4. **Next week**: Get database access for survivorship analysis
5. **Next week**: Design standalone theme quality evaluator

---

## Sources

- [LLM-as-a-Judge Guide - Evidently AI](https://www.evidentlyai.com/llm-guide/llm-as-a-judge)
- [Using LLMs for Evaluation - Cameron Wolfe](https://cameronrwolfe.substack.com/p/llm-as-a-judge)
- [Pairwise or Pointwise? - arXiv 2025](https://arxiv.org/abs/2504.14716)
- [LLMs-as-Judges Survey](https://arxiv.org/html/2412.05579v2)
- Mitch's theme generation evaluation design (internal)
- Call notes: Mitch Fruin, 3 Feb 2026
- DESNZ Pilot 1: Understanding Theme Sign Off Experience (Katie, Sep 2025)
- DWP PIP Theme Mapping Lock-In document (Katie)

# Add embedding+cluster theming as an evaluable option in themefinder

## Context

Consult's current "find themes" pipeline (`theme_generation` → `theme_condensation` → `theme_refinement`, optionally followed by the agentic `theme_clustering`) drives every LLM call through repeated prompting. We want to evaluate an alternative: generate candidate themes per batch (unchanged), embed them, cluster the embeddings, then make one LLM call per cluster to produce the final refined theme — cheaper, deterministic at the clustering step, and explicitly controllable in granularity, at the cost of blunter semantic grouping and a real risk of erasing rare/minority themes if clusters are handled naively.

This plan adds that approach to **themefinder** (`/Users/RajamanoharanG/Code/themefinder`, a separate git repo from consult, currently on `main`, clean apart from stray editor autosave files which we leave alone) as a new, optional pipeline function, and wires it into themefinder's **existing eval harness** so it can be scored head-to-head against the current pipeline using the same datasets and metrics. No consult-side (Django/batch job) changes are in scope — this is themefinder-only, for evaluation purposes.

Key finding from investigation: `evals/eval_generation.py` is the eval that actually measures find-themes quality end-to-end — it runs `theme_generation → theme_condensation → theme_refinement` and scores the final output against a ground-truth theme framework (`groundedness`, `coverage`, `specificity`, `redundancy`, all from `evals/evaluators.py`). `eval_condensation.py`/`eval_refinement.py` only test those stages in isolation with no ground truth. So the new approach must plug into the **same point** `eval_generation.py` does — after `theme_generation`, before scoring — to get a genuine apples-to-apples comparison.

## 1. New pipeline function in themefinder

**New file:** `src/themefinder/advanced_tasks/embedding_cluster.py`

This follows the existing precedent of `advanced_tasks/theme_clustering_agent.py` (a class holding the algorithm, wrapped by a thin task function in `tasks.py`).

`EmbeddingThemeClusterer` class:
- `__init__(self, llm: LLM, embedding_client: openai.AsyncOpenAI, system_prompt=CONSULTATION_SYSTEM_PROMPT, embedding_model="text-embedding-3-large", min_cluster_size=2, concurrency=10)`
- `async embed(themes: list[dict]) -> np.ndarray` — calls `embedding_client.embeddings.create(input=[f"{label}: {description}" ...], model=self.embedding_model)`. This mirrors the existing precedent in `src/themefinder/themeset_rules.py::rule_3_semantic_similarity_must_be_less_than_90pc` (same `"{label}: {description}"` input convention, same model), just async and for clustering rather than pairwise validation. No new embeddings abstraction/Protocol needed — inject the OpenAI client directly, same way `llm: LLM` is injected today.
- `cluster(embeddings: np.ndarray) -> np.ndarray` — L2-normalise, then `sklearn.cluster.HDBSCAN(min_cluster_size=self.min_cluster_size, metric="euclidean")`. Noise label `-1` is **not** dropped or force-merged: each noise point becomes its own singleton cluster. This directly avoids the minority-theme-erasure failure mode of naive clustering (e.g. k-means, which forces every point into a cluster).
- `async refine_cluster(cluster_themes: list[dict], question: str) -> RefinedTheme` — one `llm.ainvoke(prompt, output_model=RefinedTheme)` call per cluster (reusing the existing `RefinedTheme` pydantic model from `models.py` directly, no list wrapper needed since it's one cluster → one theme). New prompt (see below). Wrapped in the same `tenacity` retry style already used in `ThemeClusteringAgent.cluster_iteration` (`wait_random_exponential`, `stop_after_attempt(3)`, `reraise=True`). On exhausted retries, the cluster's source themes go to an `unprocessables` list rather than crashing the run.
- `async run(themes_df: pd.DataFrame, question: str) -> tuple[pd.DataFrame, pd.DataFrame]` — orchestrates embed → cluster → concurrent per-cluster refine (semaphore-bounded by `self.concurrency`, same pattern as `llm_batch_processor.call_llm`) → assigns sequential topic_ids → returns `(refined_df, unprocessables_df)` in the **same shape** `theme_refinement` returns (`topic_id`, `topic`, `source_topic_count` columns), so it's a drop-in replacement for `theme_condensation` + `theme_refinement` together and `theme_mapping` downstream needs no changes.
- Also returns lightweight, LLM-free cluster diagnostics (no extra API calls): `n_clusters`, `n_singletons`, `singleton_ratio` — cheap signal for whether the approach is fragmenting themes or preserving minority viewpoints, surfaced later in the eval output.

**Shared helper extraction:** `assign_sequential_topic_ids()` is currently a nested function inside `tasks.py::theme_refinement` (alphabetic A, B, ... AA, AB, ... ID assignment). Move it to a new leaf module `src/themefinder/topic_ids.py` (avoids a circular import — `tasks.py` already imports from `advanced_tasks/`, so `advanced_tasks/embedding_cluster.py` can't import from `tasks.py`). Update `theme_refinement` in `tasks.py` to import it from there too, so both paths assign IDs identically.

**New prompt** in `src/themefinder/prompts.py`: `EMBEDDING_CLUSTER_REFINEMENT` + `embedding_cluster_refinement_prompt(system_prompt, question, themes)`. Modelled on `THEME_REFINEMENT` but explicit that *all* input topics have already been determined to be similar and must be merged into exactly **one** output topic (unlike `THEME_REFINEMENT`, which doesn't guarantee a 1:1 reduction).

**New wrapper task function** in `src/themefinder/tasks.py` (exported from `__init__.py` alongside the other task functions):
```python
async def embedding_cluster_refinement(
    themes_df: pd.DataFrame,
    llm: LLM,
    embedding_client: openai.AsyncOpenAI,
    question: str,
    embedding_model: str = "text-embedding-3-large",
    min_cluster_size: int = 2,
    system_prompt: str = CONSULTATION_SYSTEM_PROMPT,
    concurrency: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
```

**Dependency:** No new package. `scikit-learn` is already a core dependency in `pyproject.toml`; bump the floor to `scikit-learn>=1.3.0` (when `sklearn.cluster.HDBSCAN` was added).

**Tests:** `tests/test_tasks.py` already has `mock_llm` (AsyncMock) and `sample_themes_df` fixtures in `conftest.py`, and tests like `test_theme_refinement`/`test_theme_clustering` mock `call_llm` or `llm.ainvoke` directly. Add `test_embedding_cluster_refinement` following that pattern: mock `embedding_client.embeddings.create` to return fixed vectors that produce a known clustering (e.g. 4 themes → 2 clusters + 1 singleton), mock `llm.ainvoke` to return a `RefinedTheme` per call, assert output shape/columns and that singleton themes pass through.

**Version bump:** `pyproject.toml` version `0.8.2` → `0.9.0` (new public API surface).

## 2. Eval harness wiring

**New file:** `evals/eval_embedding_cluster.py`, structured as a close mirror of `evals/eval_generation.py` (same `DatasetConfig(dataset=..., stage="generation")`, same Langfuse/local-fallback branching, same dataset loader `load_local_data`/`load_local_generation_data`), but the pipeline becomes:
```python
themes_df, _ = await theme_generation(responses_df, llm, question=question)
refined_df, _ = await embedding_cluster_refinement(themes_df, llm, embedding_client, question=question)
```
instead of generation → condensation → refinement. Score with the **same** evaluators `eval_generation.py` uses, imported from `evals/evaluators.py`: `create_groundedness_evaluator`, `create_coverage_evaluator`, `create_title_specificity_evaluator`, `create_redundancy_evaluator` — this is what makes the comparison apples-to-apples against the existing `generation` eval's numbers. Also surface the new `n_clusters`/`n_singletons`/`singleton_ratio` diagnostics from `EmbeddingThemeClusterer` as extra numeric scores (no LLM, no extra cost).

The `embedding_client` needed alongside `llm` is constructed the same way the existing evals construct `OpenAILLM` (same `LLM_GATEWAY_URL`/`CONSULT_EVAL_LITELLM_API_KEY` env vars, just `openai.AsyncOpenAI(base_url=..., api_key=...)` instead).

**Wire into `evals/benchmark.py`:**
- Import `evaluate_embedding_cluster` from the new module.
- Add `"embedding_cluster": evaluate_embedding_cluster` to `EVAL_FUNCS`.
- Add `"embedding_cluster"` to `evals_with_judge` (the set that decides whether `judge_llm` gets passed through in `_execute_eval`), since it also uses LLM-as-judge scoring.
- Update the `--evals` argparse help/epilog to mention it as an example (e.g. `--evals generation embedding_cluster`).

This gives a direct side-by-side run: `uv run python benchmark.py --dataset gambling_S --evals generation embedding_cluster --runs 3`, with identical cost/token/latency capture (via Langfuse) and the existing HTML report (`visualise_benchmark.py`) comparing both.

**Wire into `.github/workflows/eval.yml`:** add `embedding_cluster` to the `eval_type` `workflow_dispatch` choice list (currently `generation, mapping, condensation, refinement, all`).

No new fixture data needed — `evals/data/gambling_XS` (used by `benchmark.py --quick` and the default in all `eval_*.py` scripts) already has the `generation`-stage input/expected_output shape this reuses unchanged.

**Caveat on available data:** `gambling_XS` (100 synthetic responses × 2 question parts, with ground-truth `themes.json`/`mapping.jsonl`) is the *only* dataset actually present locally in the expected shape. `evals/data/condensation/` and `evals/data/generation/` are leftover files (`expanded_question.txt`, `framework_themes.json`, etc.), not loadable datasets despite the names. Other datasets named in `benchmark.py`'s docstring (`housing_S`, `bbc_mission_public_purposes`) only exist as remote Langfuse datasets and need `LANGFUSE_SECRET_KEY`/`LANGFUSE_PUBLIC_KEY`/`LANGFUSE_BASE_URL` to fetch — without those, local fallback has nothing to load for them. So the evaluation in this plan runs for real, end-to-end, against `gambling_XS`; running it against a larger/different consultation topic would mean either getting Langfuse access or generating a new local fixture with `evals/generate_synthetic.py`, which is out of scope here.

## Verification

1. `cd /Users/RajamanoharanG/Code/themefinder && uv run pytest tests/test_tasks.py -k embedding_cluster -v` — new unit test passes, plus run the full `tests/` suite to confirm no regression from the `assign_sequential_topic_ids` extraction.
2. `cd evals && uv run python eval_embedding_cluster.py --dataset gambling_XS` (local fallback, no Langfuse needed) — confirms the function runs end-to-end against real fixture data and produces scores.
3. `cd evals && uv run python benchmark.py --quick --evals generation embedding_cluster` — confirms both paths run side-by-side through the benchmark harness and produce a comparable summary table (this is the actual deliverable the evaluation will use).
4. Inspect `singleton_ratio` and `n_clusters` in the `gambling_XS` output (both question parts) to sanity-check the clustering isn't either collapsing everything into one cluster or leaving everything as noise. If Langfuse credentials are available, also try a larger remote dataset (e.g. `housing_S`) for a less toy-scale comparison.
5. Work happens on a new branch off `main` in the themefinder repo (not directly on `main`); I'll leave committing/pushing/PR creation to you to review first, per the cross-repo change you confirmed.

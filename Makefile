.PHONY: test test-all lint run_evals run_benchmark

test:
	uv run pytest tests/test_tasks.py tests/test_models.py tests/test_llm_batch_processor.py -v -s

test-all:
	uv run pytest tests/ -v -s

lint:
	uv run pre-commit run --all-files

run_evals:
	@echo "Running quick benchmark (1 run, gpt-4.1-mini)..."
	cd evals && uv run python benchmark.py --quick

run_benchmark:
	@echo "Running full benchmark..."
	cd evals && uv run python benchmark.py --dataset housing_S --runs 5 --provider all --judge-model gpt-4.1

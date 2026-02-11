.PHONY: run_evals run_benchmark

run_evals:
	@echo "Running quick benchmark (1 run, gpt-4.1-mini)..."
	cd evals && uv run python benchmark.py --quick

run_benchmark:
	@echo "Running full benchmark..."
	cd evals && uv run python benchmark.py --dataset housing_S --runs 5 --provider all --judge-model gpt-4.1

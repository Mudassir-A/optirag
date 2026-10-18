.PHONY: sync ingest gen api eval tune router bench dash test

sync:
	uv sync

ingest:
	uv run python -m optirag.ingest

gen:
	uv run python -m eval.generate_data all

api:
	uv run uvicorn optirag.api:app --reload

eval:
	uv run python -m eval.run_eval --mode baseline
	uv run python -m eval.run_eval --mode optimized

tune:
	uv run python -m eval.tune_threshold

router:
	uv run python -m eval.eval_router

bench:
	uv run python -m eval.benchmark

dash:
	uv run streamlit run dashboard/app.py

test:
	uv run pytest -q

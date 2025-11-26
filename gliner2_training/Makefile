install_uv:
	open https://docs.astral.sh/uv/getting-started/installation/

which-python:
	uv run which python3

create-data:
	uv run src/create_data.py --split train --out data/train.jsonl --max-rows 1000
	uv run src/create_data.py --split test --out data/eval.jsonl --max-rows 100

train:
	uv run src/train.py "fastino/gliner2-base-v1" --report-to wandb

train-help:
	uv run src/train.py --help

predict:
	uv run src/predict.py "fastino/gliner2-base-v1"
	# uv run src/predict.py "checkpoints/gliner2-20251123215102/final"

precommit:
	uv run black src --line-length 120
	uv run ruff format src

.DEFAULT_GOAL := help
.PHONY: help
help:
	@LC_ALL=C $(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null | awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' | sort | egrep -v -e '^[^[:alnum:]]' -e '^$@$$'

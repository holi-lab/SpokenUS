AUGMENT_MODEL ?= openrouter/qwen/qwen3-32b
SAMPLE_SIZE ?= 3
SPLITS ?= train,valid,test
CHUNK_SIZE ?= 10
WORKERS ?= 3
DATASETS ?= emowoz,sgd,abcd,tm2,spokenwoz
OUT_DIR ?= datasets/SpokenTOD
DATA_DIR ?= datasets
SYNTH_INPUT ?= $(OUT_DIR)/sample/train.jsonl
SYNTH_OUTPUT ?= $(OUT_DIR)/speech/train.jsonl
VLLM_OMNI_BASE_URL ?= http://127.0.0.1:8000
SYNTH_ARGS ?=

DOWNLOAD_DATASETS ?= emowoz,sgd,abcd,tm2,spokenwoz,saa
# Allow `make download DATASETS=saa` to override the download list too.
DOWNLOAD_TARGETS := $(if $(filter command line environment,$(origin DATASETS)),$(DATASETS),$(DOWNLOAD_DATASETS))

.PHONY: help sync lock augment augment-full augment-sample synthesize test lint clean download download-help

help:
	@echo "SpokenTOD Augmentation Pipeline"
	@echo ""
	@echo "Usage:"
	@echo "  make augment          - Run full augmentation pipeline"
	@echo "  make augment-sample   - Run on small sample (3 per dataset)"
	@echo "  make test             - Run unit tests"
	@echo "  make lint             - Run linter"
	@echo "  make clean            - Remove generated files"
	@echo "  make synthesize       - Synthesize audio from augmented data"
	@echo "  make download DATASETS=<name> - Download dataset archive (comma-separated list of dataset names, e.g. emowoz,sgd, errors if exists unless FORCE=1)"
	@echo "  make download-help    - Show detailed download script usage and env vars"

# base augmentation
augment:
	@clear
	@uv run python src/augment.py \
		--datasets $(DATASETS) \
		--splits $(SPLITS) \
		--data-dir $(DATA_DIR) \
		--output-dir $(OUT_DIR) \
		--chunk-size $(CHUNK_SIZE) \
		--workers $(WORKERS) \
		--model $(AUGMENT_MODEL)

augment-full:
	@echo "Starting full augmentation in background..."
	@echo "Output will be logged to nohup.out"
	nohup uv run python src/augment.py \
		--datasets $(DATASETS) \
		--splits $(SPLITS) \
		--data-dir $(DATA_DIR) \
		--output-dir $(OUT_DIR) \
		--chunk-size $(CHUNK_SIZE) \
		--workers $(WORKERS) \
		--model $(AUGMENT_MODEL) \
		> nohup.out 2>&1 &
	@echo "Process started. Monitor with: tail -f nohup.out"

augment-sample:
	@clear
	@uv run python src/augment.py \
		--datasets $(DATASETS) \
		--splits $(SPLITS) \
		--data-dir $(DATA_DIR) \
		--output-dir $(OUT_DIR)/sample \
		--sample-size $(SAMPLE_SIZE) \
		--chunk-size $(CHUNK_SIZE) \
		--workers $(WORKERS) \
		--model $(AUGMENT_MODEL)

synthesize:
	uv run --extra normalization python src/synthesize.py \
		--input-jsonl "$(SYNTH_INPUT)" --output-jsonl "$(SYNTH_OUTPUT)" \
		--reference-dir "$(DATA_DIR)/SpeechAccentArchive/recordings/recordings" \
		--base-url "$(VLLM_OMNI_BASE_URL)" $(SYNTH_ARGS)

test:
	uv run pytest tests/ -v

lint:
	uv run ruff check src/augmentation/

clean:
	rm -rf $(OUT_DIR)
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

download:
	bash scripts/download_dataset.sh $(DOWNLOAD_TARGETS)

download-help:
	bash scripts/download_dataset.sh --help

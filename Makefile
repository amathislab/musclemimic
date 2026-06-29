.PHONY: help install install-dev precommit-install check format lint test smoke ci clean

UV ?= $(shell if command -v uv >/dev/null 2>&1; then command -v uv; elif [ -x "$(HOME)/.local/bin/uv" ]; then printf '%s' "$(HOME)/.local/bin/uv"; else printf '%s' "uv"; fi)
VENV_BIN ?= .venv/bin
PYTHON ?= $(VENV_BIN)/python
PRECOMMIT ?= $(VENV_BIN)/pre-commit
PYTEST ?= $(VENV_BIN)/pytest
RUFF ?= $(VENV_BIN)/ruff
PYTEST_ARGS ?= -m "not integration"
LINT_PATHS := \
	bimanual \
	tests/unit/test_enhanced_fullbody_terminal_handler.py \
	tests/unit/test_enhanced_fullbody_terminal_handler_integration.py \
	tests/test_muscle_observations.py \
	musclemimic/core/terminal_state_handler/enhanced_fullbody.py \
	musclemimic/core/terminal_state_handler/enhanced_bimanual.py \
	musclemimic/environments/humanoids/base_bimanual.py \
	musclemimic/environments/humanoids/bimanual.py \
	musclemimic/utils/metrics.py \
	tests/unit/test_metrics.py \
	loco_mujoco/smpl/retargeting.py \
	musclemimic/core/initial_state_handler.py

help:  ## Show this help message
	@echo "MuscleMimic - Development Commands"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies
	$(UV) sync

install-dev:  ## Install dependencies including developer tools
	$(UV) sync --extra dev

precommit-install:  ## Install git pre-commit hooks
	$(PRECOMMIT) install

check: ci  ## Run the default verification suite

format:  ## Format files currently covered by repository linting
	$(RUFF) format $(LINT_PATHS)
	$(RUFF) check --fix --select I $(LINT_PATHS)

lint:  ## Run scoped lint checks without touching the rest of the repository
	$(RUFF) format --check $(LINT_PATHS)
	$(RUFF) check $(LINT_PATHS)

test:  ## Run pytest (override with PYTEST_ARGS=...)
	$(PYTEST) $(PYTEST_ARGS)

smoke:  ## Test critical package imports
	$(PYTHON) -c "from musclemimic import set_all_caches; from loco_mujoco import TaskFactory, ImitationFactory; print('Imports OK')"

ci: lint test  ## Run the CI-equivalent local checks

clean:  ## Clean cache files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true

# Berkeley SciComp Framework Makefile
# ===================================
# 
# Professional development workflow automation for the UC Berkeley
# Scientific Computing Framework, implementing best practices for
# multi-platform scientific software development.
#
# Author: Dr. Meshal Alawein (meshal@berkeley.edu)
# Institution: University of California, Berkeley
# Created: 2025
# License: MIT
#
# Copyright © 2025 Dr. Meshal Alawein — All rights reserved.

# Berkeley colors for output
BERKELEY_BLUE := \033[38;2;0;50;98m
CALIFORNIA_GOLD := \033[38;2;253;181;21m
GREEN := \033[32m
RED := \033[31m
YELLOW := \033[33m
RESET := \033[0m
BOLD := \033[1m

# Project configuration
PROJECT_NAME := Berkeley SciComp Framework
VERSION := 1.0.0
PYTHON := python3
PIP := pip3
PYTEST := pytest
BLACK := black
ISORT := isort
FLAKE8 := flake8
MYPY := mypy

# Directory structure
SRC_DIR := Python
TEST_DIR := tests
DOCS_DIR := docs
ASSETS_DIR := assets
EXAMPLES_DIR := examples
BUILD_DIR := build
DIST_DIR := dist

# File patterns
PYTHON_FILES := $(shell find $(SRC_DIR) -name "*.py")
TEST_FILES := $(shell find $(TEST_DIR) -name "*.py")
ALL_PYTHON_FILES := $(PYTHON_FILES) $(TEST_FILES)

# Default target
.DEFAULT_GOAL := help

# Berkeley banner
define BERKELEY_BANNER
$(BERKELEY_BLUE)$(BOLD)================================================================
$(PROJECT_NAME) - Development Makefile
================================================================$(RESET)

$(CALIFORNIA_GOLD)University of California, Berkeley$(RESET)
$(BERKELEY_BLUE)Dr. Meshal Alawein (meshal@berkeley.edu)$(RESET)

Version: $(VERSION)
Python: $(shell $(PYTHON) --version)
Platform: $(shell uname -s)

endef

.PHONY: help
help: ## Show this help message
	@echo "$(BERKELEY_BANNER)"
	@echo "$(BERKELEY_BLUE)Available targets:$(RESET)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(CALIFORNIA_GOLD)%-20s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(BERKELEY_BLUE)Example usage:$(RESET)"
	@echo "  make install-dev    # Set up development environment"
	@echo "  make test           # Run all tests"
	@echo "  make format         # Format code with Berkeley standards"
	@echo "  make docs           # Build documentation"
	@echo "  make demo           # Run demonstrations"

# Environment setup
.PHONY: install
install: ## Install Berkeley SciComp Framework
	@echo "$(BERKELEY_BLUE)Installing Berkeley SciComp Framework...$(RESET)"
	$(PIP) install -e .
	@echo "$(GREEN)✓ Installation complete$(RESET)"

.PHONY: install-dev
install-dev: ## Install development dependencies
	@echo "$(BERKELEY_BLUE)Setting up Berkeley development environment...$(RESET)"
	$(PIP) install -r requirements-berkeley.txt
	$(PIP) install -e .[dev,docs,jupyter]
	@$(MAKE) setup-pre-commit
	@echo "$(GREEN)✓ Development environment ready$(RESET)"

.PHONY: install-gpu
install-gpu: ## Install GPU support (CUDA)
	@echo "$(BERKELEY_BLUE)Installing GPU support...$(RESET)"
	$(PIP) install -e .[gpu]
	@echo "$(GREEN)✓ GPU support installed$(RESET)"

.PHONY: install-quantum
install-quantum: ## Install quantum computing dependencies
	@echo "$(BERKELEY_BLUE)Installing quantum computing libraries...$(RESET)"
	$(PIP) install -e .[quantum]
	@echo "$(GREEN)✓ Quantum computing support installed$(RESET)"

.PHONY: install-all
install-all: ## Install all optional dependencies
	@echo "$(BERKELEY_BLUE)Installing complete Berkeley SciComp environment...$(RESET)"
	$(PIP) install -e .[all]
	@echo "$(GREEN)✓ Complete installation finished$(RESET)"

# Code quality and formatting
.PHONY: format
format: ## Format code with Berkeley standards
	@echo "$(BERKELEY_BLUE)Formatting code with Berkeley standards...$(RESET)"
	$(ISORT) $(ALL_PYTHON_FILES)
	$(BLACK) $(ALL_PYTHON_FILES)
	@echo "$(GREEN)✓ Code formatting complete$(RESET)"

.PHONY: lint
lint: ## Run linting checks
	@echo "$(BERKELEY_BLUE)Running linting checks...$(RESET)"
	$(FLAKE8) $(ALL_PYTHON_FILES)
	@echo "$(GREEN)✓ Linting passed$(RESET)"

.PHONY: type-check
type-check: ## Run type checking
	@echo "$(BERKELEY_BLUE)Running type checks...$(RESET)"
	$(MYPY) $(SRC_DIR) --ignore-missing-imports
	@echo "$(GREEN)✓ Type checking passed$(RESET)"

.PHONY: check
check: format lint type-check ## Run all code quality checks
	@echo "$(GREEN)✓ All quality checks passed$(RESET)"

# Testing
.PHONY: test
test: ## Run all tests
	@echo "$(BERKELEY_BLUE)Running Berkeley SciComp test suite...$(RESET)"
	$(PYTEST) $(TEST_DIR) -v --tb=short
	@echo "$(GREEN)✓ All tests passed$(RESET)"

.PHONY: test-python
test-python: ## Run Python-specific tests
	@echo "$(BERKELEY_BLUE)Running Python tests...$(RESET)"
	$(PYTEST) $(TEST_DIR)/python/ -v
	@echo "$(GREEN)✓ Python tests passed$(RESET)"

.PHONY: test-coverage
test-coverage: ## Run tests with coverage report
	@echo "$(BERKELEY_BLUE)Running tests with coverage analysis...$(RESET)"
	$(PYTEST) $(TEST_DIR) --cov=$(SRC_DIR) --cov-report=html --cov-report=term
	@echo "$(GREEN)✓ Coverage report generated$(RESET)"
	@echo "$(BERKELEY_BLUE)Open htmlcov/index.html to view detailed coverage$(RESET)"

.PHONY: test-fast
test-fast: ## Run fast tests only
	@echo "$(BERKELEY_BLUE)Running fast tests...$(RESET)"
	$(PYTEST) $(TEST_DIR) -m "not slow" -v
	@echo "$(GREEN)✓ Fast tests passed$(RESET)"

# Demonstrations and examples
.PHONY: demo
demo: ## Run all demonstrations
	@echo "$(BERKELEY_BLUE)Running Berkeley SciComp demonstrations...$(RESET)"
	@$(MAKE) demo-style
	@$(MAKE) demo-quantum
	@$(MAKE) demo-ml
	@echo "$(GREEN)✓ All demonstrations complete$(RESET)"

.PHONY: demo-style
demo-style: ## Run Berkeley styling demonstration
	@echo "$(BERKELEY_BLUE)Running Berkeley visual identity demo...$(RESET)"
	$(PYTHON) $(ASSETS_DIR)/berkeley_style.py
	@echo "$(GREEN)✓ Style demonstration complete$(RESET)"

.PHONY: demo-quantum
demo-quantum: ## Run quantum physics demonstrations
	@echo "$(BERKELEY_BLUE)Running quantum physics demonstrations...$(RESET)"
	$(PYTHON) $(EXAMPLES_DIR)/python/quantum_tunneling_demo.py
	@echo "$(GREEN)✓ Quantum demonstrations complete$(RESET)"

.PHONY: demo-ml
demo-ml: ## Run machine learning physics demonstrations
	@echo "$(BERKELEY_BLUE)Running ML physics demonstrations...$(RESET)"
	$(PYTHON) $(EXAMPLES_DIR)/python/ml_physics_demo.py
	@echo "$(GREEN)✓ ML physics demonstrations complete$(RESET)"

# Documentation
.PHONY: docs
docs: ## Build documentation
	@echo "$(BERKELEY_BLUE)Building Berkeley SciComp documentation...$(RESET)"
	@if command -v sphinx-build >/dev/null 2>&1; then \
		sphinx-build -b html $(DOCS_DIR) $(BUILD_DIR)/docs; \
		echo "$(GREEN)✓ Documentation built successfully$(RESET)"; \
		echo "$(BERKELEY_BLUE)Open $(BUILD_DIR)/docs/index.html to view$(RESET)"; \
	else \
		echo "$(YELLOW)⚠ Sphinx not installed. Install with: pip install sphinx$(RESET)"; \
	fi

.PHONY: docs-serve
docs-serve: docs ## Build and serve documentation locally
	@echo "$(BERKELEY_BLUE)Serving documentation at http://localhost:8000$(RESET)"
	@cd $(BUILD_DIR)/docs && $(PYTHON) -m http.server 8000

.PHONY: docs-pdf
docs-pdf: ## Build PDF documentation
	@echo "$(BERKELEY_BLUE)Building PDF documentation...$(RESET)"
	@if command -v sphinx-build >/dev/null 2>&1; then \
		sphinx-build -b latex $(DOCS_DIR) $(BUILD_DIR)/latex; \
		cd $(BUILD_DIR)/latex && make; \
		echo "$(GREEN)✓ PDF documentation built$(RESET)"; \
	else \
		echo "$(YELLOW)⚠ Sphinx not installed$(RESET)"; \
	fi

# Building and packaging
.PHONY: build
build: clean ## Build distribution packages
	@echo "$(BERKELEY_BLUE)Building Berkeley SciComp packages...$(RESET)"
	$(PYTHON) setup.py sdist bdist_wheel
	@echo "$(GREEN)✓ Packages built successfully$(RESET)"
	@ls -la $(DIST_DIR)/

.PHONY: release
release: check test build ## Prepare for release
	@echo "$(BERKELEY_BLUE)Preparing Berkeley SciComp release...$(RESET)"
	@echo "$(GREEN)✓ Release preparation complete$(RESET)"
	@echo "$(BERKELEY_BLUE)Ready for distribution$(RESET)"

# Environment management
.PHONY: env-info
env-info: ## Show environment information
	@echo "$(BERKELEY_BLUE)Berkeley SciComp Environment Information$(RESET)"
	@echo "$(CALIFORNIA_GOLD)System:$(RESET)"
	@echo "  OS: $(shell uname -s)"
	@echo "  Architecture: $(shell uname -m)"
	@echo "  Python: $(shell $(PYTHON) --version)"
	@echo "  Pip: $(shell $(PIP) --version)"
	@echo ""
	@echo "$(CALIFORNIA_GOLD)Key Dependencies:$(RESET)"
	@$(PYTHON) -c "import sys; print(f'  NumPy: {__import__(\"numpy\").__version__}' if 'numpy' in sys.modules or __import__(\"importlib.util\").util.find_spec('numpy') else '  NumPy: Not installed')" 2>/dev/null || echo "  NumPy: Not installed"
	@$(PYTHON) -c "import sys; print(f'  SciPy: {__import__(\"scipy\").__version__}' if 'scipy' in sys.modules or __import__(\"importlib.util\").util.find_spec('scipy') else '  SciPy: Not installed')" 2>/dev/null || echo "  SciPy: Not installed"
	@$(PYTHON) -c "import sys; print(f'  Matplotlib: {__import__(\"matplotlib\").__version__}' if 'matplotlib' in sys.modules or __import__(\"importlib.util\").util.find_spec('matplotlib') else '  Matplotlib: Not installed')" 2>/dev/null || echo "  Matplotlib: Not installed"
	@$(PYTHON) -c "import sys; print(f'  TensorFlow: {__import__(\"tensorflow\").__version__}' if 'tensorflow' in sys.modules or __import__(\"importlib.util\").util.find_spec('tensorflow') else '  TensorFlow: Not installed')" 2>/dev/null || echo "  TensorFlow: Not installed"

.PHONY: setup-pre-commit
setup-pre-commit: ## Set up pre-commit hooks
	@echo "$(BERKELEY_BLUE)Setting up pre-commit hooks...$(RESET)"
	@if command -v pre-commit >/dev/null 2>&1; then \
		pre-commit install; \
		echo "$(GREEN)✓ Pre-commit hooks installed$(RESET)"; \
	else \
		echo "$(YELLOW)⚠ Pre-commit not installed. Install with: pip install pre-commit$(RESET)"; \
	fi

# Platform-specific targets
.PHONY: matlab-test
matlab-test: ## Run MATLAB tests (requires MATLAB)
	@echo "$(BERKELEY_BLUE)Running MATLAB tests...$(RESET)"
	@if command -v matlab >/dev/null 2>&1; then \
		matlab -batch "run('$(TEST_DIR)/matlab/test_heat_transfer.m'); exit"; \
		echo "$(GREEN)✓ MATLAB tests complete$(RESET)"; \
	else \
		echo "$(YELLOW)⚠ MATLAB not available$(RESET)"; \
	fi

.PHONY: mathematica-test
mathematica-test: ## Run Mathematica tests (requires Mathematica)
	@echo "$(BERKELEY_BLUE)Running Mathematica tests...$(RESET)"
	@if command -v wolframscript >/dev/null 2>&1; then \
		wolframscript -script $(TEST_DIR)/mathematica/test_symbolic_quantum.nb; \
		echo "$(GREEN)✓ Mathematica tests complete$(RESET)"; \
	else \
		echo "$(YELLOW)⚠ Mathematica not available$(RESET)"; \
	fi

# Maintenance and cleanup
.PHONY: clean
clean: ## Clean build and cache files
	@echo "$(BERKELEY_BLUE)Cleaning build artifacts...$(RESET)"
	rm -rf $(BUILD_DIR) $(DIST_DIR) *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".coverage" -delete
	rm -rf htmlcov/ .pytest_cache/ .mypy_cache/
	@echo "$(GREEN)✓ Cleanup complete$(RESET)"

.PHONY: deep-clean
deep-clean: clean ## Deep clean including virtual environments
	@echo "$(BERKELEY_BLUE)Performing deep clean...$(RESET)"
	rm -rf venv/ env/ .venv/
	$(PIP) cache purge 2>/dev/null || true
	@echo "$(GREEN)✓ Deep cleanup complete$(RESET)"

# Berkeley-specific targets
.PHONY: berkeley-banner
berkeley-banner: ## Display Berkeley SciComp banner
	@echo "$(BERKELEY_BANNER)"

.PHONY: berkeley-info
berkeley-info: ## Show Berkeley SciComp framework information
	@echo "$(BERKELEY_BLUE)$(BOLD)Berkeley SciComp Framework Information$(RESET)"
	@echo "$(CALIFORNIA_GOLD)Institution:$(RESET) University of California, Berkeley"
	@echo "$(CALIFORNIA_GOLD)Author:$(RESET) Dr. Meshal Alawein"
	@echo "$(CALIFORNIA_GOLD)Email:$(RESET) meshal@berkeley.edu"
	@echo "$(CALIFORNIA_GOLD)Version:$(RESET) $(VERSION)"
	@echo "$(CALIFORNIA_GOLD)License:$(RESET) MIT"
	@echo ""
	@echo "$(BERKELEY_BLUE)Platforms Supported:$(RESET)"
	@echo "  • Python (NumPy, SciPy, TensorFlow, JAX)"
	@echo "  • MATLAB (Engineering Toolboxes)"
	@echo "  • Mathematica (Symbolic Computation)"
	@echo ""
	@echo "$(BERKELEY_BLUE)Application Domains:$(RESET)"
	@echo "  • Quantum Physics & Quantum Computing"
	@echo "  • Machine Learning for Physics"
	@echo "  • Computational Methods & Engineering"
	@echo "  • Scientific Visualization"

.PHONY: berkeley-style
berkeley-style: ## Apply Berkeley visual identity
	@echo "$(BERKELEY_BLUE)Applying Berkeley visual identity...$(RESET)"
	$(PYTHON) $(ASSETS_DIR)/berkeley_style.py
	@echo "$(GREEN)✓ Berkeley styling applied$(RESET)"

# Continuous Integration targets
.PHONY: ci
ci: check test ## Run CI pipeline
	@echo "$(BERKELEY_BLUE)Running continuous integration pipeline...$(RESET)"
	@echo "$(GREEN)✓ CI pipeline completed successfully$(RESET)"

.PHONY: ci-full
ci-full: install-dev ci test-coverage docs ## Run full CI pipeline
	@echo "$(BERKELEY_BLUE)Running full CI pipeline with coverage and docs...$(RESET)"
	@echo "$(GREEN)✓ Full CI pipeline completed$(RESET)"

# Development workflow
.PHONY: dev
dev: install-dev berkeley-banner ## Set up complete development environment
	@echo "$(GREEN)✓ Berkeley SciComp development environment ready!$(RESET)"
	@echo ""
	@echo "$(BERKELEY_BLUE)Next steps:$(RESET)"
	@echo "  make test           # Run tests"
	@echo "  make demo           # See demonstrations"
	@echo "  make docs           # Build documentation"
	@echo "  make format         # Format code"

# Quick development cycle
.PHONY: quick
quick: format lint test-fast ## Quick development cycle (format, lint, fast tests)
	@echo "$(GREEN)✓ Quick development cycle complete$(RESET)"

# Performance targets
.PHONY: benchmark
benchmark: ## Run performance benchmarks
	@echo "$(BERKELEY_BLUE)Running performance benchmarks...$(RESET)"
	$(PYTHON) -m pytest tests/ -k "benchmark" -v
	@echo "$(GREEN)✓ Benchmarks complete$(RESET)"

.PHONY: profile
profile: ## Profile performance
	@echo "$(BERKELEY_BLUE)Running performance profiling...$(RESET)"
	$(PYTHON) -m cProfile -s cumulative examples/python/quantum_tunneling_demo.py
	@echo "$(GREEN)✓ Profiling complete$(RESET)"

# All targets that don't create files
.PHONY: all
all: dev test docs demo berkeley-info ## Run complete setup and validation
	@echo "$(GREEN)✓ Complete Berkeley SciComp setup and validation finished!$(RESET)"
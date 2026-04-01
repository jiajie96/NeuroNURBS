# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `pyproject.toml` for project metadata, dependency specification, and tool configuration (ruff, pytest, mypy)
- `.gitignore` covering Python artifacts, ML checkpoints, data directories, and IDE files
- GitHub Actions CI workflow (`ci.yml`): ruff lint + pytest on push/PR to `main`
- GitHub Actions release workflow (`release.yml`): builds wheel and sdist on version tags
- `tests/` directory with unit tests for pure utility functions (`test_utils.py`) and skipped integration tests (`test_integration.py`)
- `tests/conftest.py` stubs out heavy dependencies (OCC, wandb, diffusers) for lightweight CI testing
- `.pre-commit-config.yaml` with ruff and pre-commit-hooks
- `CONTRIBUTING.md` with development setup, workflow, and code style guidelines

### Changed
- `README.md` rewritten with badges (arXiv, Python, PyTorch, License, CI), full installation guide, repo structure, argument reference table, and citation/acknowledgement sections

## [0.1.0] — 2024-11-20

### Added
- Initial public release of NeuroNURBS
- Surface VAE and Edge VAE training (`vae.py`, `trainer.py`, `network.py`)
- Data processing pipeline for DeepCAD, ABC, and Furniture datasets (`data_process/`)
- OpenCASCADE helper utilities (`helpers/utils/`)
- NURBS construction utilities (`helpers/construct_nurbs.py`)
- Training launch scripts (`train_vae.sh`)

# Contributing to NeuroNURBS

Thank you for your interest in contributing! This document outlines the process for reporting issues, proposing changes, and submitting pull requests.

## Getting started

1. **Fork** the repository on GitHub.
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/<your-username>/NeuroNURBS.git
   cd NeuroNURBS
   ```
3. **Create a virtual environment** and install the dev dependencies:
   ```bash
   conda create -n neuronurbs-dev python=3.9.2
   conda activate neuronurbs-dev
   pip install -e ".[dev]"
   pre-commit install
   ```

## Workflow

1. **Create a branch** for your change:
   ```bash
   git checkout -b feat/my-feature   # new feature
   git checkout -b fix/bug-name      # bug fix
   ```
2. **Make your changes** and add tests where appropriate.
3. **Run the test suite** before pushing:
   ```bash
   pytest tests/ -v
   ```
4. **Commit** using a [Conventional Commits](https://www.conventionalcommits.org/) style message:
   - `feat:` new feature
   - `fix:` bug fix
   - `docs:` documentation only
   - `test:` test additions/changes
   - `refactor:` code change that doesn't add a feature or fix a bug
   - `chore:` build process / tooling changes
5. **Push** and open a Pull Request against `main`.

## Code style

This project uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting.  
Running `pre-commit install` will enforce this automatically on every commit.

To run manually:
```bash
ruff check .       # lint
ruff format .      # format
```

## Reporting issues

Please open a [GitHub Issue](https://github.com/jiajie96/NeuroNURBS/issues) and include:
- A minimal reproducible example
- Your environment (`python --version`, `torch.__version__`, OS)
- The full traceback if applicable

## Questions

For questions about the paper or method, please refer to the [arXiv paper](https://arxiv.org/abs/2411.10848) or open a Discussion on GitHub.

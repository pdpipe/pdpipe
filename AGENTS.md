# AGENTS.md - pdpipe

This file defines contributor and coding-agent rules for this repository.

## Scope and intent
- Keep changes focused, minimal, and test-backed.
- Preserve backward compatibility unless the issue or PR explicitly allows breaking changes.
- Prefer clear, deterministic behavior over implicit magic.

## Project basics
- Package: `pdpipe`
- Purpose: easy pipelines for pandas DataFrames.
- Python: CI-supported versions in this repo.
- Optional extras: `nltk`, `sklearn`.
- Primary source path: `src/pdpipe/`.

## Repository layout

```
pdpipe/
├── src/pdpipe/            # Source package
│   ├── core.py            # Pipeline and stage core abstractions
│   ├── basic_stages.py    # Core dataframe transformation stages
│   ├── col_generation.py  # Column/feature generation stages
│   ├── nltk_stages.py     # NLTK-dependent text stages
│   ├── sklearn_stages.py  # sklearn-dependent stages
│   ├── skintegrate.py     # sklearn estimator integration wrapper
│   ├── cond.py / cq.py / rq.py
│   └── ...
├── tests/                 # Pytest suite
├── docs/                  # Documentation
├── dev/                   # Developer scripts
├── .github/workflows/     # CI workflows
└── pyproject.toml         # Build + lint + test configuration
```

## Core quality requirements
- Lint/format before every commit:
  - `python -m black .`
  - `python -m flake8`
- Run relevant tests for touched code.
- Keep doctest examples valid for touched modules.

## CI workflows
- `test.yml`: main test matrix.
- `lint.yml`: flake8 checks.
- `black.yml`: formatting check.
- `npdocval.yml`: numpydoc validation.
- `release.yml`: release workflow.

## Coding expectations
- Favor explicit, readable implementations.
- Keep behavior deterministic and backward-compatible by default.
- Add regression tests for bug fixes.
- Keep optional dependencies optional; degrade gracefully when unavailable.

## PR expectations
- One logical change per PR.
- Clear summary of behavior changes and test evidence.
- Green CI before merge.

## Local overrides (optional, untracked)
- If `LOCAL_AGENTS.md` exists at repo root, treat it as additive local instructions.
- `LOCAL_AGENTS.md` should remain untracked.
- On conflicts, repository/security policy takes precedence.

## Security and secrets
- Never commit credentials or machine-specific secrets.
- Avoid weakening protections around sensitive files or env vars.

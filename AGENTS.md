# AGENTS.md - pdpipe

This file defines contributor and coding-agent rules for this repository.

## Scope and intent
- Keep changes focused, minimal, and test-backed.
- Preserve backward compatibility unless the issue or PR explicitly allows breaking changes.
- Prefer clear, deterministic behavior over implicit magic.

## Project basics
- Package: `pdpipe`
- Purpose: pandas-first pipeline stages and pipeline composition utilities.
- Python: target active support in CI (`3.10+`).
- Optional extras: `nltk`, `sklearn`.
- Primary package path: `src/pdpipe`.

## Repository layout

```
pdpipe/
├── src/pdpipe/                # Source package
│   ├── core.py                # PdPipelineStage / PdPipeline core abstractions
│   ├── basic_stages.py        # Core dataframe manipulation stages
│   ├── col_generation.py      # Feature/column generation stages
│   ├── sklearn_stages.py      # scikit-learn-integrated stages
│   ├── nltk_stages.py         # NLTK-dependent text stages
│   ├── text_stages.py         # Generic token/text stages
│   ├── cond.py                # Condition abstraction
│   ├── cq.py                  # Column qualifier abstraction
│   ├── rq.py                  # Row qualifier abstraction
│   ├── skintegrate.py         # sklearn estimator wrappers/integration
│   └── ...
├── tests/                     # Pytest suite (unit + doctest-oriented)
├── docs/                      # MkDocs/Sphinx-style docs content
├── dev/                       # Developer scripts (e.g. numpydoc validation)
├── .github/workflows/         # CI workflows
└── pyproject.toml             # Build, dependencies, lint/test tool settings
```

## Key modules

### `src/pdpipe/core.py`
Defines core contracts and mechanics:
- `PdPipelineStage` stage lifecycle and pre/post condition behavior.
- `PdPipeline` composition, fit/transform/apply orchestration.
- Stage registration/loading used by the public API.

### `src/pdpipe/basic_stages.py`
General-purpose dataframe stages such as column drop/rename/reorder, dropna, dedupe, validators, and value/row filters.

### `src/pdpipe/col_generation.py`
Column-generation transforms including `MapColVals`, `ApplyByCols`, `ApplyToRows`, `Log`, `Bin`, `OneHotEncode`, etc.

### `src/pdpipe/cond.py` and `src/pdpipe/cq.py`
Condition and column-qualification building blocks used throughout stage preconditions and dynamic column selection.

### `src/pdpipe/sklearn_stages.py` and `src/pdpipe/skintegrate.py`
sklearn-adjacent stage wrappers and pipeline/estimator bridging.

### `src/pdpipe/nltk_stages.py`
NLTK-dependent text stages; must degrade gracefully when NLTK resources are missing or unavailable.

## Core quality requirements
- Run format + lint before each commit:
  - `python -m black .`
  - `python -m flake8`
- Run relevant tests locally for touched areas; default to targeted subsets first, then broader suite as needed.

## Test suite overview
- Tests are under `tests/` and mirror module families (`basic_stages`, `col_generation`, `core`, `nltk_stages`, `sklearn_stages`, etc.).
- `pyproject.toml` pytest config includes doctests (`--doctest-modules`) and coverage.
- When fixing regressions, add or update tests near the affected module and include a direct repro case.

## CI workflows

| Workflow | Purpose |
|----------|---------|
| `test.yml` | Main pytest matrix run |
| `lint.yml` | Flake8 + related lint checks |
| `black.yml` | Black formatting check |
| `npdocval.yml` | numpydoc validation checks |
| `release.yml` | Packaging/release automation |

## Common tasks

### Adding or changing a stage
1. Implement behavior in the relevant module under `src/pdpipe/`.
2. Add/adjust tests under the matching `tests/<area>/` folder.
3. Update docs/examples/docstrings if user-facing behavior changed.
4. Run `black`, `flake8`, and targeted pytest before commit.

### Changing stage preconditions/errors
1. Keep exception types and messages stable unless a compatibility change is intended.
2. Add tests for both pass and fail paths.
3. Verify doctest examples still execute as documented.

## Coding expectations
- Favor explicit, readable logic over clever compactness.
- Preserve deterministic output ordering where applicable.
- Avoid silent behavior changes; document and test intentional changes.
- Keep optional dependencies optional: imports should fail gracefully when extras are not installed.

## PR expectations
- Keep PR descriptions explicit: what changed, why, and test evidence.
- Prefer one logical change per PR.
- Ensure CI is green before merge.

## Local overrides (optional, untracked)
- If `LOCAL_AGENTS.md` exists at repo root, treat it as additive local instructions.
- On conflicts:
  - Security and repository policy rules take precedence.
  - Then `LOCAL_AGENTS.md` may refine local workflow and tool routing.
- Never commit machine-specific paths, personal tokens, or local MCP server names into tracked docs.

## Security and secrets
- Never commit secrets or credentials.
- Do not weaken protections around sensitive files or environment variables.
- Avoid introducing network calls in tests unless explicitly marked and isolated.

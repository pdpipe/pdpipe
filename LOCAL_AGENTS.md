# LOCAL_AGENTS.md

Machine-local instructions for this repository. This file is intended to be untracked.

## MCP routing (local)
- For git MCP operations in this repo, prefer `mcp__git_pdpipe__*` when available.
- If `mcp__git_pdpipe__*` is unavailable in a given session, use generic git MCP only if it is configured to this repo.
- If git MCP is unavailable or path-restricted, use local git CLI as fallback.

## GitHub routing (local)
- Use GitHub MCP for remote operations whenever possible.
- Default repo for GitHub MCP calls from this workspace:
  - `owner="pdpipe"`
  - `repo="pdpipe"`

## Workflow preferences (local)
- Prefer MCP tools over shell when both are available and equivalent.
- Keep environment/version manager files local-only unless explicitly requested to commit.
- Run `black` and `flake8` before every commit.
- Add tests for behavior changes and update docs when user-facing behavior changes.

## Example local override
Use this as a template and edit it for your machine/session:

```md
# LOCAL_AGENTS.md (example)

## MCP routing (local)
- Prefer `mcp__git_pdpipe__*`.

## GitHub routing (local)
- owner="pdpipe"
- repo="pdpipe"

## Workflow preferences (local)
- Run `python -m black .` and `python -m flake8` before each commit.
- Prefer targeted pytest first, then broader suite.
```

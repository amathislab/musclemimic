# Contributing

Thanks for your interest in contributing to MuscleMimic! We welcome bug fixes, new features, documentation improvements, tests, and general maintenance contributions.

Checkout [our roadmap](ROADMAP.md] for a list of planned features. Contributions that help us reach those goals are especially welcome.

We especially encourage contributions from people from backgrounds that are underrepresented in open-source software. If you want to discuss an idea before opening a pull request, feel free to open an issue or email the team!

If you are new to GitHub, the [GitHub Guides](https://docs.github.com/en) are a great place to start.

## Development Setup

This repository targets Python 3.11 and uses `uv` for environment management.

```bash
make install-dev
make precommit-install
```

Optional extras:

- `uv sync --extra dev --extra cuda` for Linux x86_64 CUDA setups
- `uv sync --extra dev --extra smpl --extra gmr` for full SMPL/GMR retargeting workflows

## Local Checks

Use the `Makefile` targets for the default developer workflow:

```bash
make format
make lint
make test
make ci
```

Developer tooling is expected inside `.venv/bin`. `make lint`, `make test`, and `make smoke` use those local executables directly instead of spawning `uv run` each time.

By default, `make test` runs:

```bash
.venv/bin/pytest -m "not integration"
```

If you need a different subset, override `PYTEST_ARGS`:

```bash
make test PYTEST_ARGS='tests/unit/test_ppo_config.py -q'
```

## Pre-commit Scope

`pre-commit` is intentionally scoped to a curated subset of files in `.pre-commit-config.yaml`.

`make lint` and `make format` follow that same scoped set of Python paths. They are intentionally not whole-repository checks today.

This is a migration guardrail, not an accident. Please do not remove or broadly expand the `files:` allowlist in unrelated pull requests. Large formatting-only churn has previously broken working flows and made review difficult.

If you want to expand `pre-commit` coverage:

1. Do it in a dedicated cleanup PR.
2. Confirm the newly covered paths pass `ruff` and `pytest` as appropriate.
3. Keep the diff reviewable and separate from functional changes.

## Pull Requests

Before opening a PR:

1. Run `make ci`.
2. Update tests for behavior changes.
3. Update `README.md` or other docs when user-facing commands or workflows change.
4. Keep refactors separate from behavior changes when practical.

PRs should include:

- A short summary of the change
- The motivation or linked issue
- The validation you ran locally
- Screenshots or logs when they help reviewers

## Issues

Use the GitHub issue templates for bug reports and feature requests.

For bug reports, include:

- The command you ran
- Your platform and Python version
- Whether you used CPU, CUDA, MuJoCo, SMPL, or GMR extras
- The full traceback or failure log
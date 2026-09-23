# MuscleMimic Development Guidelines

This file applies to the entire repository. Add a nested `AGENTS.md` only when a
subsystem needs stricter or more specific guidance.

## Project priorities

MuscleMimic is a JAX-based research codebase for muscle-actuated motion
imitation. Preserve these properties when changing it:

1. Correct physical and numerical behavior.
2. Reproducible training, evaluation, and checkpoint resume.
3. JAX/MJX/Warp performance without obscuring correctness.
4. Clear module ownership and stable public interfaces.
5. Small, reviewable changes with focused tests.

Do not trade correctness or reproducibility for a cleaner-looking abstraction.
When a refactor changes results, configuration semantics, compilation behavior,
or checkpoint compatibility, treat it as a behavior change and document it.

## Repository architecture

Use this target dependency direction:

```text
fullbody/ and bimanual/ entry points, scripts, analyses, examples
                              |
                              v
              runner, evaluation, and visualization
                              |
                              v
        algorithms, rl_core, environments, and musclemimic/core
                              |
                              v
       loco_mujoco simulation, trajectory, and data primitives
```

Responsibilities:

- `loco_mujoco/`: generic simulation, control, observation, trajectory, dataset,
  terrain, and retargeting primitives. It should not depend on application-level
  `musclemimic` training or runner code.
- `musclemimic/core/`: MuscleMimic-specific MJX state, goals, rewards, terminal
  handlers, and wrappers.
- `musclemimic/environments/`: environment and robot composition. Do not put
  training loops, logging backends, or CLI behavior here.
- `musclemimic/algorithms/` and `musclemimic/rl_core/`: learning algorithms,
  networks, optimizers, and rollout data structures. Keep environment-specific
  policy out of reusable algorithm code where practical.
- `musclemimic/runner/` and `musclemimic/evaluation/`: host-side orchestration,
  lifecycle, logging, checkpoint coordination, and evaluation.
- `musclemimic/viewer/` and `musclemimic/web_viewer/`: presentation and
  interactive tooling. Core simulation and algorithm modules must not depend on
  viewers.
- `fullbody/` and `bimanual/`: thin Hydra entry points and experiment
  configuration, not alternate implementations of shared behavior.
- `tests/`: behavior and regression coverage. Mirror source boundaries where it
  makes tests easier to find.

Some imports currently run against this direction. Treat them as existing
technical debt, not precedent. Do not introduce a new reverse dependency to
complete an unrelated task. If a boundary must be crossed, expose a small public
interface or move the genuinely shared concept to a neutral lower layer.

## Module and API design

- Depend on public interfaces. Do not import a leading-underscore symbol from
  another module.
- Keep `__init__.py` exports intentional and small. Avoid new wildcard imports
  and compatibility re-export chains.
- Before changing a public function, class, config key, registry name, or
  serialized field, search its uses in source, configs, tests, scripts, and
  examples.
- Preserve public signatures and configuration/checkpoint compatibility by
  default. If a break is necessary, provide migration guidance and, when
  practical, a deprecation path.
- Prefer cohesive modules over generic dumping grounds such as `utils.py`.
  Separate orchestration, domain computation, I/O, and presentation when they
  evolve independently.
- Extract one boundary at a time. First characterize existing behavior with a
  test; then move code without mixing in algorithmic changes.
- Avoid new dependencies unless the standard library or an existing dependency
  cannot reasonably solve the problem.
- Comments should explain intent, constraints, units, or non-obvious edge cases,
  not restate the code.
- Add type annotations to new public interfaces. Put types in signatures rather
  than repeating them in docstrings.
- Document units and array shape conventions for physical quantities at public
  boundaries, for example `[m]`, `[rad]`, `[N]`, and
  `[num_envs, num_joints]`.

## Documentation writing

All documentation uses concise, factual, analytical prose and active voice.
State claims directly. Replace rhetorical negation, contrastive pairings,
subjective qualifiers, and explanatory padding with functional descriptions.
Use no em dash or en dash. Reserve the ASCII hyphen for compound words,
hyphenation, literal commands, configuration keys, and mathematical syntax.
Rewrite clause breaks with a period, semicolon, colon, comma, or parentheses.
Keep formatting minimal. Use prose and code blocks when they carry technical
meaning. Omit tables, icons, emoji, decorative separators, marketing-style
headings, and visual padding.

## Python project quality and code clarity

- Goal: adopt project-grade quality norms common in high-volume open-source Python ML
  libraries and reproducible research codebases.

  Baseline references we align with:
  - PEP 8 (style, naming, imports, line breaks, comments, and code layout).
  - PEP 257 (docstring conventions).
  - Google Python Style Guide (type annotations, naming, docstring style).
  - TensorFlow code style and mature ML project contributor guidance
    (lint/type-aware review habits).
  - pytest conventions for test import/discovery and test fixtures.
  - Reproducibility checklists (Nature ML checklist, Pineau v2.0, Yahoo recipe).

- Constant and magic-value policy:
  - Avoid non-obvious magic numbers in logic.
  - Use named constants for domain thresholds, schedule points, buffer sizes, and
    algorithmic defaults; use `UPPER_CASE` for shared module-level constants.
  - If a literal must stay inline, annotate with intent/provenance in a short
    comment (paper source, ablation choice, or hard API contract).
  - Keep constants close to where they are consumed; when shared across modules,
    promote them to explicit, owned config/constants modules.
- Intent-first naming for structured payloads:
  - Prefer explicit keys / fields over positional tuples for logs, rollout metrics,
    and trainer outputs.
  - Keep metric/logging schema explicit and stable; changes to schema should be
    covered by tests and migration notes.
- Comments and docstrings:
  - Document units and scale for numeric operations in public boundaries:
    e.g. `[m]`, `[rad]`, `[N]`, `[step]`, `[num_envs, num_joints]`.
  - Keep comments concise, truthful, and updated when behavior changes.
  - Use comments for branch intent, fallback reason, invariants, or numerical edge
    constraints (not to restate obvious code).
- Readability and maintainability:
  - Prefer simple code over over-engineered abstractions; if a helper exists,
    it should reduce complexity and test burden, not add indirection.
  - Keep public interfaces explicit; add type annotations to new public functions.
- Testing and reproducibility integration:
  - Prioritize tests for behavior changes; keep tests deterministic and isolated.
  - Use `tmp_path`/`tmp_path_factory` for filesystem-dependent tests in new code.
  - Reproducibility artifacts should document: data provenance, preprocessing
    choices, random-seed flow, hyperparameters, training dynamics, and
    environment/commit references for experiments.

## JAX, MJX, and numerical code

- Functions transformed by `jax.jit`, `jax.vmap`, or `jax.grad` must be pure.
  Pass state explicitly and return updated state. Keep file I/O, logging,
  mutation, and other host side effects outside transformed functions.
- Thread PRNG keys explicitly and split them before independent uses. Never
  silently reuse a key unless identical samples are intentional and documented.
- Do not convert tracers to NumPy arrays or use traced values in Python boolean
  control flow. Use JAX arrays and `jax.lax` control-flow primitives, or make
  truly static values explicit.
- Keep array shapes and pytree structure stable across compiled steps. Be
  deliberate about static arguments; high-cardinality static values can cause
  repeated compilation.
- Place JIT boundaries around cohesive outer computations. Do not scatter new
  `jax.jit` decorators through low-level helpers without measuring compilation
  and runtime effects.
- Keep dtype, precision, device placement, and host-device transfers explicit
  when they affect results or performance.
- Preserve backend behavior across supported MuJoCo, MJX, and Warp paths. A fix
  for one backend must not silently route another backend through incompatible
  code.
- Use `jax.debug.print` for runtime values inside transformed code. Remove
  temporary debug output before finishing the change.
- Warm up compiled functions and call `.block_until_ready()` when timing JAX
  execution. Report compilation time separately from steady-state runtime.
- Numerical changes need tests with justified tolerances. Do not loosen a
  tolerance merely to hide a regression or backend mismatch.

## Configuration, data, and checkpoints

- Treat Hydra configuration keys and defaults as user-facing interfaces. Keep
  entry-point configs declarative and put shared behavior in Python modules.
- Do not embed machine-specific dataset, asset, cache, or output paths in source
  or committed configs.
- Preserve deterministic seed flow through environment reset, trajectory
  sampling, policy action, and evaluation.
- Treat checkpoint layout and resume behavior as compatibility-sensitive. Test
  fresh start, save/load, and resume when changing state or manifests.
- Do not commit generated datasets, model assets, checkpoints, recordings,
  caches, or experiment output.

## Development workflow

This repository uses Python 3.11, `uv`, Ruff, and pytest. Follow
`CONTRIBUTING.md` for setup and the authoritative workflow.

```bash
make install-dev
make precommit-install

# Fast feedback: run the narrowest relevant test first.
make test PYTEST_ARGS='tests/unit/test_relevant_area.py -q'

# Repository checks.
make lint
make test
make ci
```

- Developer tools are expected in `.venv/bin`; use the Make targets rather than
  inventing a parallel environment workflow.
- `make lint` intentionally covers a curated migration subset. Do not broaden
  that scope or reformat unrelated files in a feature or bug-fix change.
- Mark tests requiring external resources, special assets, or hardware as
  integration tests. Keep unit tests local and deterministic.
- For a regression fix, verify that the new test fails without the fix and
  passes with it.
- Test both eager and JIT-transformed behavior when tracing is relevant.
  Disabling JIT alone is not proof that compiled execution works.
- Prefer targeted tests during iteration; run `make ci` before handing off a
  change when the local environment supports it.
- If a required check cannot run because of missing hardware, credentials,
  gated data, or optional dependencies, state exactly what was not run and why.

## Change hygiene

- Inspect the working tree before editing and preserve unrelated user changes.
- Keep refactors separate from behavior changes when practical.
- Avoid drive-by formatting, renaming, or cleanup outside the requested scope.
- Update tests and user-facing documentation when commands, configs, public
  behavior, or expected outputs change.
- In the handoff, summarize the behavior changed, list validation performed,
  and call out remaining risks or checks not run.

## Reference guidance

These sources informed this file; this repository's rules above are
authoritative:
- [JAX: Stateful computations](https://docs.jax.dev/en/latest/stateful-computations.html)
- [JAX: Errors](https://docs.jax.dev/en/latest/errors.html)
- [JAX: Benchmarking](https://docs.jax.dev/en/latest/benchmarking.html)
- [PEP 8: Style Guide for Python Code](https://peps.python.org/pep-0008/)
- [PEP 257: Docstring Conventions](https://peps.python.org/pep-0257/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [TensorFlow Code Style Guide](https://www.tensorflow.org/community/contribute/code_style)
- [PyTorch Lightning Contributing Guide](https://lightning.ai/docs/pytorch/stable/generated/CONTRIBUTING.html)
- [pytest Good Integration Practices](https://pytest.org/en/8.0.x/explanation/goodpractices.html)
- [Yahoo ML Reproducibility Guidelines](https://github.com/yahoo/ml-reproducibility-guidelines)
- [Machine Learning Checklist (Nature)](https://www.nature.com/documents/machine-learning-checklist.pdf)
- [Machine Learning Reproducibility Checklist v2.0](https://www.cs.mcgill.ca/~jpineau/ReproducibilityChecklist-v2.0.pdf)

# Repository Guidelines

## Project Structure & Module Organization
- Keep production code in `src/` with clear subpackages (e.g., `src/core`, `src/policies`, `src/utils`); place CLI or scripts in `scripts/`.
- Store experiments, notebooks, and reproducible configs in `experiments/` with dated folders; keep raw data in `data/` and avoid committing large artifacts.
- Add tests under `tests/` mirroring the `src/` layout; co-locate fixtures in `tests/fixtures/`.
- Document decisions and runnable examples in `docs/` or `README` files within each module.

## Build, Test, and Development Commands
- `conda activate rl`: activate the project environment before any install/run steps.
- `pip install -e .[dev]`: install the package in editable mode with dev extras; keep the list in `requirements-dev.txt` or `pyproject.toml`.
- `make format`: run formatting (e.g., Black + isort); add this target if missing.
- `make lint`: run static checks (e.g., Ruff/flake8, mypy); ensure new targets live in the Makefile.
- `make test`: execute the full test suite (pytest) and emit coverage; keep it green before pushing.

## Coding Style & Naming Conventions
- Prefer Python 3.11+ with 4-space indentation, `snake_case` for functions/variables, `PascalCase` for classes, and `SCREAMING_SNAKE_CASE` for constants.
- Require type hints; enable strict mypy settings (`--strict`) on new modules.
- Use docstrings on public functions/classes; keep module headers minimal and focused on purpose.
- Run formatters before commits; avoid hand-editing generated files.

## Testing Guidelines
- Use pytest with files named `test_*.py`; mirror package structure so imports stay simple.
- Add fixtures in `tests/fixtures/` and prefer factory/helpers over hardcoded values.
- Target >90% coverage for new modules; mark slow/integration tests with `@pytest.mark.slow` and gate them behind an opt-in flag.
- When fixing a bug, add a regression test that fails before the fix.

## Commit & Pull Request Guidelines
- Use Conventional Commits (`feat:`, `fix:`, `docs:`, `chore:`, `refactor:`, `test:`); keep subjects ≤72 chars and bodies wrapped at 100.
- One logical change per commit; rebase locally to keep history linear.
- PRs should include: summary of changes, rationale/linked issue, test evidence (`make test` output), and any follow-up tasks.
- When altering behavior or APIs, update relevant docs and add migration notes if needed.

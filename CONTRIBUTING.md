# Contributing Guide

Thanks for your interest in improving this project.

## Development setup

1. Fork and clone the repository.
2. Create and activate a virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Copy environment template if needed:

```bash
copy .env.example .env
```

## Local quality checks

Before opening a pull request, run:

```bash
python -m compileall app.py chatbot tests
python -m pytest -q
```

## Contribution workflow

1. Create a feature branch from `main`.
2. Keep changes focused and small when possible.
3. Add or update tests for behavior changes.
4. Update docs when behavior, config, or architecture changes.
5. Open a pull request using the PR template.

## Commit message style

This repository follows a conventional style:

- `feat:` new functionality
- `fix:` bug fix
- `docs:` documentation-only updates
- `test:` tests only
- `refactor:` code structure improvements without behavior changes

Example:

```text
feat: add OCR fallback for scanned PDFs
```

## Pull request checklist

- [ ] Tests pass locally
- [ ] New behavior is covered by tests
- [ ] README/docs are updated when needed
- [ ] No secrets or credentials are committed

## Reporting issues

Use GitHub Issues for bugs and feature requests.
For security vulnerabilities, follow `SECURITY.md`.

# MIKE IO QA Strategy

**Audience:** MIKE IO developers and DHI Head of Product QA.
**Scope:** Current practice. This document codifies what we do today, not what we aspire to do.

## Context

MIKE IO began as a research project and has only been on stable funding for around two years (since 2024). The practices below reflect a deliberate maturation from research-grade towards production-grade, but the project is not yet uniformly Level 3 across the [Northstar Quality Maturity Framework](#) dimensions. We describe the current state honestly; gaps are named rather than hidden.

## What MIKE IO is, for QA purposes

MIKE IO is a high-level Python abstraction on top of [`mikecore`](https://github.com/DHI/mikecore-python), which provides the low-level bindings to the DFS and EUM C libraries. Numerical correctness of model output is the responsibility of the MIKE engines upstream of `mikecore`; correctness of low-level file I/O is `mikecore`'s concern. **MIKE IO's QA focus is the abstraction layer itself: that the high-level API (`Dataset`, `DataArray`, geometry classes, generic operations) correctly preserves and exposes the data underneath.** This framing determines what we test, and what we deliberately do not.

## Practices by dimension

### 1. Quality Strategy & Governance

Accountability for MIKE IO sits with the Product Owner of Python Core libraries. Day-to-day quality decisions are made by the core maintainers via PR review. Architectural decisions are captured in [`adr/`](adr/) (8 ADRs to date); project conventions live in [`CONTRIBUTING.md`](CONTRIBUTING.md), [`CLAUDE.md`](CLAUDE.md), and [`CONTEXT.md`](CONTEXT.md). There is no separate QA-owner role; quality is the maintainers' joint responsibility, gated by CI and code review.

### 2. Test Process & SDLC Integration

Tests run in CI on every push and pull request to `main` ([`.github/workflows/full_test.yml`](.github/workflows/full_test.yml)) across Python 3.10 and 3.14. Scheduled runs add Windows coverage every three days. `CONTRIBUTING.md` requires a failing test before a bug fix is merged. Defects are tracked in [GitHub Issues](https://github.com/DHI/mikeio/issues) as the single source of truth; triage is informal and severity-driven (data-corruption issues jump the queue) rather than label- or SLA-driven. We do not track defect trends.

### 3. Unit Test Automation

The suite contains 832 tests across 28 top-level test files, exercising approximately 18,000 lines of source code. Line coverage is 95% (measured on-demand via `just coverage`; not surfaced on PRs or tracked over time). Tests, `ruff` linting, and `mypy` type checking are required CI gates; any failure blocks merge. Coverage is not itself a numerical gate.

### 4. Integration Testing

Dedicated suites assert cross-format invariants: [`test_roundtrip.py`](tests/test_roundtrip.py) (read → write → read preserves data and metadata), [`test_consistency.py`](tests/test_consistency.py), [`test_read_consistency.py`](tests/test_read_consistency.py), and [`test_integration.py`](tests/test_integration.py) (end-to-end workflows on real test files). These run on every PR. Roundtrip preservation is the central property we guard.

### 5. End-to-End Testing

Workflow-level testing is provided by the integration suites above, plus notebook execution tests in [`tests/notebooks/`](tests/notebooks/), run by [`notebooks_test.yml`](.github/workflows/notebooks_test.yml). The notebooks double as living documentation and as E2E coverage of typical advisory/research workflows. We do not maintain a separate UI/E2E framework; there is no UI.

### 6. Test Data & Environment Management

All test data (125 files, approximately 21 MB) is committed under [`tests/testdata/`](tests/testdata/) and versioned with the code. There is no external test-data store and no separate provisioning step. CI environments are reproducible via the `uv.lock` lockfile; a [`.devcontainer`](.devcontainer/) configuration is provided for local development.

### 7. Domain Validation & Scientific Quality

**MIKE IO is a high-level abstraction on top of `mikecore`; it does not perform model computations.** Numerical correctness of physics belongs to the MIKE engines; correctness of low-level binary I/O belongs to `mikecore`. MIKE IO's validation strategy is therefore preservation-based: the roundtrip and consistency suites (Dimension 4) assert that data and metadata pass through the abstraction without corruption or silent reinterpretation. We do not maintain numerical reference baselines, and we are explicit that this is by design given the layer we occupy.

### 8. Non-Functional Testing

Performance tests live under [`tests/performance/`](tests/performance/) and run weekly on Ubuntu and Windows ([`perf_test.yml`](.github/workflows/perf_test.yml)). They serve as a **regression tripwire**: they will fail loudly on a crash or hang. We do not currently review their output, compare durations across releases, or gate on performance thresholds. There is no formal reliability or scalability testing.

### 9. Skills, Roles & Organization

The core maintainer team is small. Merge and release authority rest with the core maintainers. External contributors interact through PRs reviewed by a maintainer; CI enforces the same checks for all contributors. Onboarding is documentation-led: [`CONTRIBUTING.md`](CONTRIBUTING.md) is the entry point, [`CONTEXT.md`](CONTEXT.md) and [`CLAUDE.md`](CLAUDE.md) capture project conventions, and [`adr/`](adr/) explains why decisions were made. There is no formal training or certification programme; the documentation is the programme.

### 10. Metrics & Quality Insights

**No quality metrics are tracked over time.** Coverage, open-issue counts, time-to-fix, flake rate, and PyPI download statistics are all observable on demand but are not collected into a dashboard or used systematically in decision-making. The operational signal we rely on is the binary state of CI on `main`. This is a Level-1 dimension and we note it as such.

## Release process

MIKE IO follows a trunk-based model: **`main` is always in a releasable state.** A release is therefore a communication event, not a stability event. Versions in development carry a `.dev0` suffix (e.g. `3.2.0.dev0` in [`pyproject.toml`](pyproject.toml)); cutting a release means removing the suffix, tagging, and creating a GitHub release, which automatically triggers PyPI publish via [`python-publish.yml`](.github/workflows/python-publish.yml). There is no pre-release sign-off, smoke test, or checklist beyond CI being green. Recent cadence: v3.0.0 (Dec 2024), v3.0.1, v3.1.0 (Mar 2026).

## What we deliberately do not do

To make the strategy legible, the following are *intentional* omissions today, not oversights:

- No numerical reference baselines for scientific outputs (see Dimension 7 — outside our layer).
- No performance trend tracking (Dimension 8 — tripwire only).
- No quality-metrics dashboard (Dimension 10).
- No formal QA-owner role separate from the Product Owner and maintainers (Dimension 1).
- No pre-release sign-off ritual (released continuously from `main`).

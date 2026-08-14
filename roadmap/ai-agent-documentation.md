# Making MIKE IO documentation excellent for AI agents

An execution plan. Not a product-roadmap feature — this file sits outside `roadmap/features/`
on purpose, so it does not appear in the generated public roadmap.

## Goal

An AI agent — Claude Code, Cursor, Copilot, a ChatGPT session, anything — asked to write a
MIKE IO script for advisory work should produce code that **runs correctly on the first
attempt and is idiomatic**, without the user having to paste documentation into the prompt.

## Scope

**In scope: consumer agents.** Agents writing MIKE IO *user scripts* — the advisory/consulting
use case where the script is the deliverable. This is where the leverage is: every script
written by an agent anywhere benefits, not just work inside this repo.

**Out of scope: contributor agents.** Agents working *inside* this repository are already
served by `CLAUDE.md`, `CONTEXT.md`, and `adr/`. No work items here target them.

**Also out of scope:**

- An MCP server. Everything it would serve is static text that the two channels below already
  deliver, and it would be a service to operate.
- Per-page `.md` alternates on the docs site — subsumed by `llms-full.txt`.
- Getting indexed by Context7/DeepWiki. A downstream benefit of better public docs, not a
  work item.

## Delivery channels

Two, in priority order:

1. **Shipped in the wheel** — `AGENTS.md` inside the installed package. Always present,
   version-matched to the installed code, no network required. Advisory work often happens on
   locked-down machines; this is the only channel that cannot be unreachable or out of sync.
2. **Published on the docs site** — the same content at a stable URL, plus `llms-full.txt`,
   for agents that fetch.

Both are generated from one source so they cannot diverge.

## How success is measured

Every work item that changes agent-facing content cites a failure it fixes, observed in the
eval. Two tiers of scoring:

1. **Execution gate (primary, deterministic).** Does the generated script run against
   `tests/testdata/` and produce the expected result? Catches hallucinated methods, wrong
   argument names, wrong axis order — the failures that actually matter. No model involved.
2. **Rubric judge (secondary).** `claude-haiku-4-5` scoring against an explicit checklist for
   what execution cannot see: idiomatic style, `ds["name"]` over attribute access, no removed
   or deprecated API, no reinvention of something `generic.py` already provides. Explicit
   checklist, never open-ended "is this good" — the judge has no strong prior on mikeio
   idioms. Batch API for offline scoring; a full sweep costs cents.

The eval does **not** run in CI: it is non-deterministic, costs money, and needs an API key.
It is invoked explicitly via `just agent-eval` and its results are committed as dated
scorecards, so the baseline is visible in git and improvements are provable across releases.
It is deliberately *not* a pytest test that skips without an API key — an invisible skip
reporting green is worse than no test at all.

## Decisions already made

Do not re-litigate these while executing:

| Decision | Choice |
|---|---|
| Audience | Consumer agents only |
| Channels | Wheel + docs site; no MCP server |
| Wheel artifact | A single `AGENTS.md` as package data — not a directory, not a `mikeio.agent_guide()` function, not a separate distribution |
| Eval approach | Evidence-driven: baseline first, then fix what it shows |
| Eval scoring | Execution gate primary, Haiku rubric judge secondary |
| Eval location | In-repo, excluded from CI, `just agent-eval`, committed scorecards |
| Anti-rot | Every fenced Python block in `AGENTS.md` is executed by a test in CI |
| Site output | Frontmatter descriptions + `llms-full.txt` + published `AGENTS.md` |

## Open questions, to be answered by evidence during execution

- **How long may `AGENTS.md` be?** A working cap of ~400 lines / ~5k tokens is the starting
  hypothesis, not a decision. Too long and agents skim it or it crowds the user's actual task
  out of the context window; too short and it omits the fix. Decided by item 6.
- **Does a test-data index earn its place in `AGENTS.md`?** It only helps agents that have the
  repo or the sdist, so its value is unproven. Decided by item 6.

---

# Work items

Ordered. Each is self-contained with a concrete acceptance check. Items within a phase are
independent of each other unless stated.

Work on a branch and open a PR — never commit to `main`.

## Phase 0 — Foundations

These are independent of the eval, cheap, and useful on their own. Do them first.

### 1. Close the docs-CI gap for source-only changes

**Problem.** `.github/workflows/docs.yml` triggers on pull requests with
`paths: ['docs/**']` only, while the push trigger includes `src/mikeio/**`. A PR that changes
only `src/mikeio/**` therefore does not rebuild the docs. An API change that breaks a
user-guide code block passes review, lands on `main`, and only then fails the docs build —
after the breakage is merged.

**Change.** Add `src/mikeio/**` to the `pull_request` `paths` list in
`.github/workflows/docs.yml`, matching the push trigger.

**Acceptance.** A PR touching only a file under `src/mikeio/` triggers the Documentation
workflow.

**Why it belongs in an agent-documentation plan.** Quarto executes the code in the user guide
on every build. That build *is* the correctness guarantee for the largest body of
agent-readable example code in the project — and today it does not gate PRs.

### 2. Add frontmatter descriptions to every documentation page

**Problem.** `docs/generate_llms_txt.py` already reads a `description:` field from YAML
frontmatter and emits it into `llms.txt`, but only 2 of 19 pages have one (`user-guide/pfs.qmd`
and `examples/Time-interpolation.qmd`). The published `llms.txt` is therefore a bare list of
titles and URLs: an agent cannot tell which page answers its question without fetching all of
them.

**Change.** Add a one-line `description:` to the frontmatter of each page listed in the
`_quarto.yml` sidebar and examples sections. Write them for an agent deciding whether to
fetch: state what the page covers and when it is the right page, not what it is called.

Pages missing a description:

```
docs/design.qmd
docs/examples/index.qmd
docs/examples/dfs0/index.qmd
docs/examples/dfs2/index.qmd
docs/examples/dfsu/index.qmd
docs/user-guide/data-structures.md
docs/user-guide/dataarray.qmd
docs/user-guide/dataset.qmd
docs/user-guide/dfs0.qmd
docs/user-guide/dfs1.qmd
docs/user-guide/dfs2.qmd
docs/user-guide/dfsu.qmd
docs/user-guide/eum.qmd
docs/user-guide/generic.qmd
docs/user-guide/getting-started.qmd
docs/user-guide/mesh.qmd
docs/user-guide/statistics.qmd
```

**Acceptance.** `just docs` succeeds, and every non-API entry in the generated
`docs/_site/llms.txt` carries a description.

## Phase 1 — Establish the baseline

### 3. Build the agent-eval harness

**Create `tests/agent_eval/`** containing:

- `tasks/` — one markdown file per task. Each holds the prompt given to the agent, and the
  assertion the resulting script must satisfy. Tasks are realistic advisory work, drawn from
  what the user guide and examples suggest people actually do:
  read a dfsu and extract a point time series to dfs0; resample to hourly means; subset by
  area or by element; compute statistics over a period; read a dfs2 and write a modified copy;
  read a mesh and inspect its geometry; use `generic.concat` / `generic.extract`;
  read a PFS file and change a parameter. Aim for 15–25 tasks spanning dfs0/1/2/3, dfsu, mesh
  and pfs.
- `rubric.md` — the explicit checklist the judge scores against. Concrete, checkable items
  only.
- `run.py` — the runner. For each task: prompt the agent under test, write the script to a
  temp dir, execute it against `tests/testdata/`, record pass/fail and any traceback, then
  send the script plus the rubric to `claude-haiku-4-5` for the style pass. Emit a scorecard.
- `results/` — dated scorecards, committed.
- `README.md` — how to run it, what the numbers mean, and how to add a task.

**Wire it up:**

- Add an `agent-eval` recipe to the `justfile`.
- Ensure the directory is excluded from default pytest collection — extend the existing
  `addopts` in `pytest.ini`, which already ignores `tests/performance/` and `tests/notebooks/`.
- The runner reads its API key from the environment and **fails loudly** if unset. It never
  skips silently.

**Design constraints.**

- The generating agent must be given **no repo context** — the point is to measure what an
  agent knows from the model's priors plus whatever the package ships, which is the situation
  a real user is in.
- Pin the judge model id (`claude-haiku-4-5`) in config so scorecards are comparable over time.
- The execution gate is the primary number. Report both, but never let a good judge score
  paper over a failing execution gate.

**Acceptance.** `just agent-eval` runs end to end and writes a scorecard; running it with no
API key fails with a clear message rather than skipping.

### 4. Run and commit the baseline

Run the eval against the current release. Commit the scorecard to
`tests/agent_eval/results/`, and write a short failure inventory alongside it: every distinct
failure mode, how often it occurred, and which task exposed it.

**This inventory is the input to Phase 2.** Nothing in `AGENTS.md` gets written that is not
traceable to a line in it.

**Acceptance.** A committed scorecard and a failure inventory naming each distinct failure mode.

## Phase 2 — The artifact

### 5. Write `src/mikeio/AGENTS.md`

`AGENTS.md` is the emerging cross-agent convention, so agents that sweep a project tree or a
virtualenv recognise the filename without being told. That is the primary discovery route.

**Content**, in this order, each item justified by the Phase 1 inventory:

- **Mental model** — compressed from `CONTEXT.md`: file → Dataset → DataArray, with time axis
  and geometry. Prevents an agent inventing an xarray-shaped API.
- **Canonical recipes** — runnable snippets for the tasks that recur. Copy-paste-correct, each
  one executed by the test in item 7.
- **Footguns** — the things the baseline showed agents getting wrong. Expected candidates,
  to be confirmed against the inventory rather than assumed: `ds["name"]` in preference to
  dynamic attribute access; the time dimension being present even on single-snapshot files;
  1-based indices in the file format versus 0-based in Python; when to use `mikeio.open()`
  rather than `mikeio.read()`.
- **Where to look next** — a pointer table from task to user-guide page, so the agent knows
  what it has *not* been told.
- **Test-data index** — only if item 6 shows it earns its place.

**Constraints.**

- It is not a second user guide. The user guide is ~2100 lines and already exists; anything
  that is reference material is a pointer, not prose.
- Every snippet must be executable against committed test data. If it cannot be, it does not
  go in.
- Prefer showing the correct idiom over prohibiting the wrong one.

**Acceptance.** Each section traces to a failure in the Phase 1 inventory; the file is within
the budget set by item 6.

### 6. Decide the length budget and the test-data index, by measurement

Run the eval against two or three variants of `AGENTS.md` — for instance a minimal version
(mental model + footguns only), the full draft, and the full draft plus the test-data index.
Compare execution-gate pass rates.

Adopt the smallest variant that captures the gain. Record the numbers and the decision in the
eval README, so the budget is a measured constraint rather than a guess that ossifies.

**Acceptance.** A documented comparison and a stated budget.

### 7. Make the recipes executable and CI-enforced

**Create `tests/test_agents_md.py`.** Parse the fenced ```python blocks out of
`src/mikeio/AGENTS.md`, execute each against `tests/testdata/`, and fail on any exception.

This is the anti-rot mechanism. A recipe that breaks when the API changes fails CI on the PR
that broke it — the same PR that then has to fix the documentation. It also gives agents the
property they most need: the snippets are provably correct.

It is deterministic and fast, so unlike the eval it belongs in CI and runs with the normal
suite.

**Acceptance.** The test collects every fenced Python block in `AGENTS.md` and passes;
deliberately breaking a snippet makes it fail.

### 8. Ship `AGENTS.md` in the wheel and point to it from the package

**Package data.** The build backend is `uv_build`. Configure it so `src/mikeio/AGENTS.md` is
included in both the wheel and the sdist, landing at `site-packages/mikeio/AGENTS.md`.

Verify by building and inspecting the artifacts rather than trusting the configuration:

```bash
uv build
python -m zipfile -l dist/mikeio-*.whl | grep AGENTS.md
```

**Docstring pointer.** Add one line to the module docstring in `src/mikeio/__init__.py`:
that agent guidance lives in `AGENTS.md` alongside the package. This surfaces in
`help(mikeio)`, IDE hover, and the generated API reference — a second discovery route for
agents that introspect rather than grep.

**Acceptance.** The file is present in a built wheel at `mikeio/AGENTS.md`, and
`help(mikeio)` mentions it.

## Phase 3 — Publish

### 9. Serve the same content from the docs site

Extend `docs/generate_llms_txt.py` to additionally emit:

- **`llms-full.txt`** — `AGENTS.md` followed by the full user guide as plain markdown. One
  fetch gives an agent everything, instead of crawling 19 HTML pages and stripping Quarto
  markup. It will be large (tens of thousands of tokens); that is fine for a fetchable
  resource, and it is exactly why `AGENTS.md` stays small and separate.
- **`AGENTS.md`** copied to the site root at a stable URL, generated from
  `src/mikeio/AGENTS.md` so the two copies cannot drift.

Link both from `llms.txt`, and add an `<link rel="alternate">` for `llms-full.txt` in
`docs/_llms-head.html` next to the existing `llms.txt` entry.

**Acceptance.** After `just docs`, `docs/_site/` contains `llms-full.txt` and `AGENTS.md`;
the latter is byte-identical to `src/mikeio/AGENTS.md`; `llms.txt` links to both.

## Phase 4 — Verify and keep it alive

### 10. Re-run the eval and publish the delta

Run the eval against the branch with everything above in place. Commit the scorecard next to
the baseline and state the delta plainly: which failure modes are gone, which remain, and
which are not fixable by documentation.

**Acceptance.** A second committed scorecard and a written comparison against the baseline.

### 11. Institutionalise it

- Add a short section to `CONTRIBUTING.md`: what `AGENTS.md` is for, that its snippets are
  tested, and how to run the eval.
- Add "run `just agent-eval` and commit the scorecard" to the release checklist, so drift is
  caught at a natural cadence rather than never.
- Consider an ADR recording the decision to ship agent-facing documentation inside the
  package, and why the wheel is the primary channel.

**Acceptance.** A contributor can find out how to run the eval without asking.

---

## Dependency order

```
1 ─┐
2 ─┼─→ (independent, ship any time)
   │
3 ─→ 4 ─→ 5 ─→ 6 ─→ 7 ─→ 8 ─→ 9 ─→ 10 ─→ 11
```

Items 1 and 2 have no dependencies and deliver value immediately. Everything from 5 onwards
depends on the Phase 1 baseline: writing `AGENTS.md` before measuring means fixing imagined
failures instead of real ones.

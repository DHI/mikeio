---
name: release
description: Cut a MIKE IO release, or decide what version bump a set of changes deserves. Gathers evidence of breaking and additive changes since the last tag, recommends major/minor/patch, drafts the GitHub release notes, then prepares the release PR, publishes the release that uploads to PyPI, and bumps main to the next .dev0. Use for "cut a release", "release 3.4.0", "what bump is this", "is this breaking", "draft release notes".
---

# Releasing MIKE IO

Publishing a GitHub release runs `.github/workflows/python-publish.yml`, which
tests on Linux and Windows and uploads to PyPI. PyPI never accepts the same
version twice, so every step before publishing has to be right.

**You do not choose the version bump.** Gather evidence, recommend, then ask.
The maintainer decides — see step 4.

## 1. Establish the baseline

```bash
git fetch --tags
git switch main && git pull --ff-only
git describe --tags --abbrev=0                  # the tag to compare against
git log --oneline $(git describe --tags --abbrev=0)..HEAD
grep -n '^version' pyproject.toml
```

Releases are cut from `main`. The version on `main` is normally `<next>.dev0`,
but the dev bump has been missed before, leaving it at the last released
version. Don't trust it as the next version: step 6 sets the version the
maintainer chose in step 4, whatever `main` says.

## 2. Run the mechanical check

```bash
uv run griffe check mikeio -s src -a $(git describe --tags --abbrev=0)
```

griffe compares the public Python API at the tag with the working tree:
removed objects, removed or renamed parameters, changed defaults. It reports
public names defined inside private modules (`src/mikeio/dfs/_dfs1.py`), which
is what users import through `mikeio`. Read its output, don't just relay it;
no output means no signature changes, not no breaking changes.

## 3. Inspect what griffe cannot see

**Behaviour.** Same signature, different result: a corrected interpolation, a
changed default inside a function body, a different delete-value mapping. A bug
fix that changes returned numbers breaks anyone who relied on the old numbers.
Read the commit log and the PR descriptions — each PR states its Merge Danger
and Blast Radius.

**Exceptions.** A call that used to raise `AssertionError` and now raises
`TypeError` breaks code that caught the old type.

**Written files.** Anything that changes the bytes `to_dfs` writes — header
fields, custom blocks, item order, delete values — can change how MIKE Zero
and the engines read the file.

**Dependency and Python floors.** Raised under
[SPEC 0](https://scientific-python.org/specs/spec-0000/) (see `CONTRIBUTING.md`).
They can stop an install; list them as breaking.

```bash
git diff $(git describe --tags --abbrev=0)..HEAD -- pyproject.toml
```

**Deprecations due for removal.** Code marked `remove in v4.0` goes only in a
major release:

```bash
grep -rn "remove in v" src/mikeio
```

## 4. Present evidence, then ask

Summarise: breaking changes, additive changes, what you checked by hand and
what you concluded, and anything you could not verify. Recommend a bump with
the reasoning, then ask with `AskUserQuestion` (major/minor/patch, your
recommendation first and labelled). Do not continue on a bump you picked.

Semver applies. Raising a dependency floor has gone into minor releases
(3.2.0, 3.3.0); removing or renaming documented API needs a major.

## 5. Draft the release notes

Follow the structure of the [v3.3.0 notes](https://github.com/DHI/mikeio/releases/tag/v3.3.0):

```
## What's new in v<VERSION>

### New features
### Breaking changes
### Documentation

**Full changelog:** https://github.com/DHI/mikeio/compare/v<PREVIOUS>...v<VERSION>
```

- Link the PR for each entry and say what a user can now do, with the case
  that motivated it.
- A breaking change says why it was made, not only what changed, and what to
  write instead.
- Only changes that affect typical user workflows count as breaking. Renaming
  an internal class or removing a stale export does not.
- Don't present something that already existed as new.
- Leave out sections with no entries.

Write the notes to a scratch file and show them to the maintainer before step 6.

## 6. Release PR

```bash
git switch -c release/<VERSION> origin/main
```

- `pyproject.toml`: set `version` to the chosen version, e.g. `3.4.0`.
- `CITATION.cff`: set `version` and `date-released`. It has fallen behind
  before; check both fields.

```bash
just check
```

Commit as `Release <VERSION>`, push, and open the PR. It merges last, after any
other PR meant for this release.

## 7. Publish

After the release PR is merged:

```bash
git switch main && git pull --ff-only
gh release create v<VERSION> --target main --title "Release <VERSION>" --notes-file <notes.md>
gh run watch $(gh run list --workflow python-publish.yml --limit 1 --json databaseId -q '.[0].databaseId')
```

`gh release create` publishes immediately, which starts the upload. Confirm
the version reached PyPI:

```bash
curl -s https://pypi.org/pypi/mikeio/json | python -c "import json,sys; print(json.load(sys.stdin)['info']['version'])"
```

## 8. Bump main to the next dev version

```bash
git switch -c dev-version origin/main
```

Set `pyproject.toml` to the next minor with `.dev0` (`3.4.0` → `3.5.0.dev0`),
commit as `Dev version`, and open the PR. Without it, code installed from
GitHub reports the released version number. The release is not finished until
this PR is open; don't end the session before it is.

## If the publish fails

The workflow tests before it uploads, so a failure in `test` means nothing
reached PyPI. Fix on `main`, then re-run with `gh workflow run python-publish.yml`
(it also takes `workflow_dispatch`) — but that builds `main`, not the tag, so
only do it if `main` has no other changes since the tag; otherwise delete the
release and tag and recreate them on the fixed commit:

```bash
gh release delete v<VERSION> --cleanup-tag
```

If the upload succeeded and the release is wrong, the version is spent: fix it
in a patch release.

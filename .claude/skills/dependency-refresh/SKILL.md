---
name: dependency-refresh
description: >-
  Run a dependency & version refresh for this repo: Python libraries in
  pyproject.toml/uv.lock, Docker base images, GitHub Actions pins, the uv CLI
  version (CI + README + Dockerfiles), the supported Python range, and the
  infra images (postgres/mongo/LangFuse) used by compose and the smoke tests.
  Use when asked to "update dependencies", "bump versions", "do a dependency
  refresh", or on the scheduled monthly dependency-refresh run. Self-contained
  playbook; prior state (deferred majors, cooldown dates) lives in the previous
  refresh PR, which Step 0 tells you how to find.
---

# Dependency & Version Refresh

This skill is the complete playbook. Running state — what moved each round and
which majors are deferred — lives in **refresh PR descriptions**, not in a file.

Supporting references in this skill's directory — read them at the step that
needs them, not up front:

- `references/coupling-constraints.md` — project-specific coupling constraints
  and gotchas learned in past refreshes (langchain⇄langgraph lockstep,
  checkpointer majors, Actions runtime deprecations, uv quirks, …). Skim the
  headings before triage; read fully when a resolver conflict or CI surprise hits.
- `references/live-e2e.md` — the live end-to-end verification recipe
  (fake-model HTTP ladder + Streamlit browser tests).

## Conventions: PRs are the state record

Every refresh produces exactly one PR that both delivers the changes and
records the state the *next* refresh starts from:

- **Branch:** `claude/dependency-refresh-YYYY-MM-DD`
- **Title:** `chore(deps): dependency refresh YYYY-MM-DD`
- **Body sections** (the template):

```markdown
## What moved
<pin raises, lock-only movement, Actions/uv CLI, infra images, base images —
grouped, with one-line release-note checks for anything 0.x or major>

## Deferred majors & cooldowns
<carry this table forward from the previous refresh PR: drop rows that landed
(note them under "What moved"), keep rows still deferred, add newly discovered
majors. Every key-dependency major gets a row the round it is first seen.>

| Upgrade | From → To | Major released | Eligible from | Notes / ROI |
| --- | --- | --- | --- | --- |

## Verification
<checks run and their results, including which smoke targets were re-run and
which were deliberately skipped and why>
```

Never stack extra state elsewhere: no journal files, no backlog docs. If a
learning is *reusable knowledge* (a coupling constraint, a gotcha), record it in
`references/coupling-constraints.md` in the same PR — that file is knowledge,
not state.

## Step 0 — Recover state from the last refresh PR

Before touching any pin, find the most recent refresh PR and read its body:

1. Search PRs (state: all, newest first) with query
   `"dependency refresh" in:title` scoped to this repo.
2. Fallback: search by head-branch prefix `head:claude/dependency-refresh`.
3. Last resort: `git log` on `main` for commits touching `uv.lock`, then look up
   their PRs.

From the latest one (or two, if the latest was narrow), extract: the deferred
majors table, anything "deliberately held back", and verification caveats
(e.g. "MongoDB untested"). That table seeds this round's triage; carry it
forward per the template above.

## Cooldown policy

Two layers, one mechanism where possible:

**1. Global 7-day resolver cooldown (enforced by uv).** `pyproject.toml` has:

```toml
[tool.uv]
exclude-newer = "7 days"
```

Every resolution (`uv lock`, `uv lock --upgrade`, `uv sync` re-locks) only
considers releases at least 7 days old, so a compromised or broken
fresh release can't enter the lock during its highest-risk window. This
automatically limits pin raises too: raising a `~=` pin to a version younger
than 7 days makes the resolve fail with an `exclude-newer` hint — that is the
policy working, not an error to route around. Hold that bump for the next
round instead of overriding.

**Security exception (the only sanctioned override):** if a release fixes a
vulnerability that affects this repo (Dependabot PR, advisory), take it
immediately regardless of age — add a temporary per-package override with a
comment, and remove it in the next refresh once the release is older than the
global cooldown:

```toml
[tool.uv.exclude-newer-package]
somepkg = "0 days"  # security fix CVE-XXXX; remove once >7 days old
```

Cooldown mechanics to know: the span is recorded in `uv.lock`
(`exclude-newer-span`), re-locks are stable, and it needs the repo's pinned uv
version — see the uv entries in `references/coupling-constraints.md`.

**2. Three-month major cooldown for key dependencies (enforced by process).**
Key dependencies are the ones where a bad major is expensive or sticky:

- **Infra images:** `postgres`, `mongo` (compose/smoke tests) — majors can
  change on-disk formats.
- **App platform:** `streamlit`, `fastapi`, `pydantic`.
- **Core agent stack:** `langgraph` + `langchain` and the coupled
  `langgraph-checkpoint-*` packages.

A new major of a key dependency is not adopted until **at least 3 months after
its X.0.0 release date**, and then only in its own dedicated PR (never inside a
safe-bumps round). The round that first sees the new major adds a row to the
deferred table with the major's release date and the computed "eligible from"
date; subsequent rounds carry the row until it lands or is rejected. Eligible ≠
automatic: when the date arrives, do the normal major triage (below) before
landing it. Non-key majors need no 3-month wait — just ordinary triage.

## Where versions live

| What | File |
| --- | --- |
| Runtime deps + version pins | `pyproject.toml` → `[project] dependencies` |
| Dev tooling (ruff, pyrefly, pytest, …) | `pyproject.toml` → `[dependency-groups] dev` |
| Minimal client/Streamlit deps (a subset, **kept in sync** with the main list) | `pyproject.toml` → `[dependency-groups] client` |
| Fully resolved versions (source of truth for what installs) | `uv.lock` |
| Resolver cooldown | `pyproject.toml` → `[tool.uv] exclude-newer` |
| Lint/format target Python | `pyproject.toml` → `[tool.ruff] target-version` |
| CI test matrix | `.github/workflows/test.yml` (`python-version`) |
| Container base image | `docker/Dockerfile.app`, `docker/Dockerfile.service` |
| Supported Python range + classifiers | `pyproject.toml` → `requires-python`, `classifiers` |
| GitHub Actions versions (`actions/checkout`, `setup-python`, `setup-uv`, `docker/*`, `codecov-action`, …) | `.github/workflows/*.yml` (`uses:`) |
| `uv` CLI version — **four places, keep in sync** | `.github/workflows/test.yml` (both `setup-uv` steps' `version:`), `docker/Dockerfile.app` + `docker/Dockerfile.service` (`pip install uv==`), `README.md` quickstart (`curl .../uv/<version>/install.sh`) |
| Infra images used by compose + smoke tests | `compose.yaml` (`postgres:` tag), `docker/compose.mongo.yaml` (`mongo:` tag), `scripts/smoke_test.sh` (`LANGFUSE_REF`) |

## The workflow

1. **Recover state** (Step 0 above).
2. **Survey.** Compare resolved versions in `uv.lock` against the latest on
   PyPI (`https://pypi.org/pypi/<package>/json` → `info.version`,
   `info.requires_dist`, `info.requires_python`; each release's file
   `upload_time_iso_8601` gives the age for cooldown checks). Also survey the
   non-Python surfaces from the table above: Actions pins, uv CLI, Docker base
   images, infra image tags, `LANGFUSE_REF`.
3. **Triage** each candidate (principles below), honoring both cooldowns.
   New key-dependency majors get a deferred-table row, not a bump.
4. **Apply safe bumps** by editing pins in `pyproject.toml`. Preserve the
   existing style: `~=` for app deps, `>=` floors for loosely-pinned libs.
5. **Re-resolve:** `uv lock --upgrade`. Read conflicts carefully — they name
   the exact transitive constraint (or cooldown) blocking a bump; check
   `references/coupling-constraints.md` for known ones.
6. **Reconcile pins to the lock.** Where the resolver picked higher than the
   written pin (common for `>=` floors), raise the written pin to match — after
   which `uv lock` must report no changes. **Sanity-check each reconciled floor
   against what actual dependents require** before writing it down (grep the
   package's block in `uv.lock` for reverse-dep `specifier`s) — otherwise a
   floor nothing needs can block a future upgrade.
7. **Sync + static verify:** `uv sync --frozen`, then `uv run ruff check .`,
   `uv run ruff format --check .`, `uv run pyrefly check`, `uv run pytest`.
8. **Live e2e:** follow `references/live-e2e.md` (fake-model HTTP ladder;
   Streamlit browser smoke when the UI stack moved). If a bump touches
   checkpointers, AG-UI, or LangFuse, also run the relevant
   `scripts/smoke_test.sh` target (see the smoke-test skill; needs the Docker
   daemon — `sudo dockerd &` if it isn't running).
9. **PR:** branch, title, and body per the conventions above. Never push to
   main. Record new reusable gotchas in the coupling reference in the same PR.

## Triage principles

- **Minor/patch bumps** within a major are generally safe — batch them.
- **Major bumps** (and pre-1.0 `0.x` minors, which don't promise SemVer
  stability): assess the actual code/behavior change against how the repo
  *uses* the package, weigh ROI, and hold non-trivial ones for their own PR.
  Grep for the import first — often the repo touches only a tiny, stable part
  of the API.
- **Key-dependency majors** additionally wait out the 3-month cooldown.
- **Version-coupled packages** move in lockstep and a bump in one drags others
  — the known couplings are in `references/coupling-constraints.md`; the
  resolver will find the rest.
- **Transitive-only deps** (nothing in the repo imports them) are lower risk
  for our code; their risk lives in whatever consumes them (e.g.
  Streamlit ↔ pandas/pyarrow). Don't add them to `[project] dependencies`
  just to pin them.

## Python version policy

Follow CPython's cycle: a new minor each October, ~5 years of support (~18
months bugfix, then security-only). Keep the declared range
(`requires-python`), classifiers, CI matrix, ruff `target-version`, and Docker
base images mutually consistent — all listed in the table above. Adopt a new
minor once the dependency stack resolves and passes on it; drop a minor when
it nears security-EOL or a needed dependency drops it first (numpy is usually
the leading indicator). Verify support claims on a real interpreter of that
version (`uv python install 3.X`, then the full check ladder) — and use a
stable patch release, not an `rc` (see the coupling reference).

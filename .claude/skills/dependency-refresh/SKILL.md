---
name: dependency-refresh
description: >-
  Run a dependency & version refresh for this repo: Python libraries in
  pyproject.toml/uv.lock, Docker base images, GitHub Actions pins, the uv CLI
  version (CI + README + Dockerfiles), and the supported Python range. Use when
  asked to "update dependencies", "bump versions", "do a dependency refresh",
  or on the scheduled quarterly dependency-refresh run. The full playbook lives
  in docs/Dependency_Upgrades.md — this skill is the entry point that routes
  you there.
---

# Dependency & Version Refresh

The complete, battle-tested playbook is **`docs/Dependency_Upgrades.md`**. Read it
in full before touching any pin — it contains the workflow recipe, the table of
every place versions live (including Docker images, GitHub Actions, and the uv CLI
pin that must stay in sync across three files), triage principles for majors vs.
safe bumps, and a long list of coupling constraints learned the hard way
(langchain⇄langgraph lockstep, checkpointer majors, Actions Node-runtime
deprecations, over-tight floors from past pin-reconciliation, …).

## Operating rules

1. **The doc is the source of truth.** Follow its recipe (survey → triage → safe
   bumps → `uv lock --upgrade` → reconcile pins → verify → live e2e). Don't
   improvise a different workflow.
2. **Safe bumps in one PR; each deferred major gets a row in the doc's backlog
   table** with a From → To and an ROI rationale.
3. **Update the doc as part of the refresh.** New coupling constraints, gotchas,
   or landed majors get recorded in `docs/Dependency_Upgrades.md` in the same PR —
   that's how the next refresh stays cheap.
4. **Verify before pushing:** `uv run ruff check` + `format --check`,
   `uv run mypy src`, `uv run pytest`, and the fake-model live e2e from the doc.
   If the bump touches checkpointers, AG-UI, or LangFuse, also run the relevant
   `scripts/smoke_test.sh` target (see the smoke-test skill; requires the Docker
   daemon — start it with `sudo dockerd &` if it isn't running).
5. **Open a PR; never push to main.** The PR description should separate "what
   moved" from "what was deliberately held back and why".

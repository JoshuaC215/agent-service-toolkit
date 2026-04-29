# Dual-Track Agent Platform Plan

## Intent

Build this repository as a Python-first fast path for LangGraph agents inside the AI operating system, while keeping Skill Companion and DWH Readiness as production-grade app packs.

## North Star

- Fast path: a new agent can be scaffolded, run, and deployed quickly.
- Product path: Skill and DWH remain first-class, tested, and releasable.
- Clear boundaries: core runtime is stable, app-specific logic is isolated.

## Target Repository Model

- `src/core/`: shared runtime concerns (settings, model resolution, logging, common helpers).
- `src/service/`: generic API surface (`invoke`, `stream`, `history`, `info`).
- `src/agents/core/`: canonical baseline agents and shared registration primitives.
- `src/agents/packs/skill/`: Skill Companion agent pack.
- `src/agents/packs/dwh/`: DWH Readiness agent pack.
- `src/apps/skill_companion/`: Skill Streamlit entrypoint.
- `src/apps/dwh_readiness/`: DWH Streamlit entrypoint.

## Guardrails

- Core modules must not import app-pack modules directly.
- Agent discovery/registration must be manifest-driven.
- Every supported pack must ship with tests and smoke checks.
- Experimental packs must be labeled and isolated from default UX.

## Work Packages

### WP1 - Stabilization Baseline

Goal: establish a reproducible, green baseline before structural changes.

Deliverables:
- Clean branch from stable base.
- Baseline test report committed to PR description.
- Debt snapshot (P0/P1/P2) linked to tasks.

Acceptance:
- `pytest` green on baseline branch.
- No unreviewed local-only artifacts tracked.

### WP2 - Registry and Contract Layer

Goal: decouple agent/app registration from hardcoded imports.

Deliverables:
- `AgentPack` interface (id, description, stability, loaders).
- registry module for loading built-in packs.
- `/info` metadata extended with `track` and `stability` tags.

Acceptance:
- Existing routes behave unchanged for current agents.
- tests cover registry discovery and info serialization.

### WP3 - Pack Migration (Skill + DWH)

Goal: move product logic into pack folders without changing user-visible behavior.

Deliverables:
- Skill companion pack folder with agent + app entrypoint.
- DWH pack folder with agent + app entrypoint.
- compatibility aliases for old import paths.

Acceptance:
- old entrypoints still function.
- new entrypoints function.
- no regression in core service tests.

### WP4 - Fast Path Developer Experience

Goal: make Python-only agent onboarding obvious and short.

Deliverables:
- `scripts/new_agent_pack.py` scaffold helper (optional but recommended).
- docs quickstart: create, run, test, deploy a minimal pack.
- one tiny reference pack as "hello-pack".

Acceptance:
- new contributor can create and run a pack in less than 20 minutes.

### WP5 - CI and Release Tracks

Goal: enforce quality independently for core and app packs.

Deliverables:
- CI matrix:
  - core contract tests
  - skill pack tests
  - dwh pack tests
  - smoke tests for run_service + streamlit launch
- release checklist with rollback notes per track.

Acceptance:
- failing pack test does not hide core stability signal.
- release notes include track scope and risk.

## Milestones

- M1: baseline + registry skeleton merged.
- M2: skill and dwh migrated to packs with compatibility layer.
- M3: docs/CI split complete and fast path published.

## Proposed PR Sequence

1. PR-A: baseline + plan + non-functional scaffolding.
2. PR-B: registry contract + metadata tags + tests.
3. PR-C: skill pack migration + compatibility aliases.
4. PR-D: dwh pack migration + compatibility aliases.
5. PR-E: docs refresh + CI matrix + release checklist.

## Risks and Mitigations

- Risk: hidden coupling between service and current agent imports.
  - Mitigation: keep compatibility aliases until M3.
- Risk: large refactor destabilizes deploy pipeline.
  - Mitigation: PR slicing and smoke checks per PR.
- Risk: ambiguity returns in docs.
  - Mitigation: dual-lane README with explicit "Fast Path" and "Products" sections.

## Success Metrics

- green CI on core and both packs for two consecutive release candidates.
- no open TODO comments in core product paths.
- reduced onboarding time for new agent pack setup.
- improved clarity: users can answer "Is this core, product, or experimental?" for each pack.

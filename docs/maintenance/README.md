# Maintenance Routines

This directory holds the executable playbooks for the automated maintenance
[Routines](https://code.claude.com/docs/en/routines) on this repo. Each Routine is
created at [claude.ai/code/routines](https://claude.ai/code/routines) with a short
prompt that just points at one of these files; the file is the source of truth for
what the run does. Change behavior by editing the file (via PR) — the Routine
itself only names the file.

These files are maintainer scaffolding: `.github/workflows/template-cleanup.yml`
strips the whole `docs/maintenance/` directory from downstream template clones.

## The Routines

| Routine | Playbook | Schedule | Writes to GitHub? |
| --- | --- | --- | --- |
| Daily sentinel | `Daily_Sentinel.md` | Daily | No — read-only urgency check |
| Maintenance run | `Weekly_Maintenance_Run.md` | Sunday 10:00 UTC, biweekly (parity-gated) | Draft-only, plus pre-authorized stale closes |
| CI follow-through | `CI_Follow_Through_Run.md` | Sunday `45 10 * * 0` UTC | Pushes fixes to the run's own `claude/` PR branches only |

## Design principle: no self-wake-ups

A scheduled wake-up back into a *running* session is unreliable — the cloud
environment is [reclaimed after a short idle period](https://code.claude.com/docs/en/claude-code-on-the-web),
so the resume may never arrive. So no playbook ever ends its turn to "wait": each
run goes straight through to its result. Work that must happen after a delay runs
as its **own** fresh Routine instead. That is why CI follow-through — cleaning up
CI on the PRs the maintenance run opens — is a separate Routine that fires ~45 min
later, rather than the maintenance run waiting on CI. A fresh session inherits the
repo's committed guardrails (`.claude/settings.json` deny rules; `claude/`-only
branch pushes) automatically.

## Routine prompts

Create each Routine with the matching prompt below. Give the CI follow-through
Routine **"Allow unrestricted branch pushes" OFF**.

**Daily sentinel:**

> You are the daily sentinel for JoshuaC215/agent-service-toolkit. Read docs/maintenance/Daily_Sentinel.md in the cloned repo and follow it exactly. You are READ-ONLY on GitHub: never post, close, label, or push. Check the last 24h of GitHub activity, CI on main, and the live app smoke test. If nothing clears the document's urgency bar, end with the single line "Sentinel: no urgent activity." and nothing else. Only produce a real alert message when something genuinely meets the bar.

**Maintenance run:**

> You are the scheduled maintenance run for JoshuaC215/agent-service-toolkit. Read docs/maintenance/Weekly_Maintenance_Run.md in the cloned repo and execute it exactly. Start with the parity gate: anchor Sunday 2026-07-12; on an off-week, follow the minimal instructions provided, then end. On an on-week, run the phases as the document specifies — community triage and everything else is DRAFT-ONLY except the stale-sweep closes, which are pre-authorized; code-changing phases open PRs and never push to main — and finish with the single structured digest the document defines.

**CI follow-through:**

> You are the CI-follow-through run for JoshuaC215/agent-service-toolkit. Read docs/maintenance/CI_Follow_Through_Run.md in the cloned repo and follow it exactly. React to CI results only — never to PR comments or reviews. Get the maintenance run's freshly-opened claude/ PRs to green where a quick fix is within reach, pushing only to those PRs' existing claude/ branches; never merge, never push to main. End with a short report of each PR's resulting CI state.

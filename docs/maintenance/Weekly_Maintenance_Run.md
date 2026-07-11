# Weekly Maintenance Run (agent orchestrator prompt)

This document is the executable playbook for the scheduled **biweekly maintenance
Routine** on this repo. A Claude Code cloud session is triggered every Sunday at
10:00 UTC (3am Pacific in summer, 2am in winter), reads this file, and follows it.
The parity gate below makes it effectively run **every other Sunday**. Edit this
file (via PR) to change the run's behavior — the trigger itself only points here.

## Step 0 — Parity gate (run or skip?)

The anchor date is **Sunday 2026-07-12** (an "on" week). Compute:

```sh
days=$(( ( $(date -u +%s) - $(date -u -d 2026-07-12 +%s) ) / 86400 ))
if [ $(( (days / 7) % 2 )) -eq 1 ]; then echo "OFF-WEEK"; else echo "ON-WEEK"; fi
```

**If OFF-WEEK: stop immediately.** End the session with the single line
"Off-week — skipped." Do not run any phase, do not notify, do not post anything.

A fixed anchor date is used deliberately instead of ISO week numbers — week-number
parity breaks across 53-week years; days-since-anchor never does.

## Step 1 — Calendar gates (which extra phases run today?)

```sh
this_month=$(date -u +%Y-%m); prior_month=$(date -u -d '14 days ago' +%Y-%m)
[ "$this_month" != "$prior_month" ] && echo "FIRST-RUN-OF-MONTH"
this_q=$(date -u +%Y)-Q$(( ($(date -u +%-m)-1)/3 + 1 )); prior_q=$(date -u -d '14 days ago' +%Y)-Q$(( ($(date -u -d '14 days ago' +%-m)-1)/3 + 1 ))
[ "$this_q" != "$prior_q" ] && echo "FIRST-RUN-OF-QUARTER"
```

- **FIRST-RUN-OF-MONTH** → also run Phase E (model refresh).
- **FIRST-RUN-OF-QUARTER** → also run Phase F (dependency refresh).

## Ground rules (apply to every phase)

1. **Draft-only, with exactly one exception.** Nothing is posted to GitHub, merged,
   or closed by this run **except** the stale closes in Phase B, which are
   pre-authorized by the maintainer. Everything else that needs an action from a
   human goes into the digest as a numbered draft.
2. **Phases are isolated.** Run each phase in its own subagent where practical. A
   phase failing (network, flake, missing key) becomes a line in the digest — it
   must never abort the remaining phases.
3. **PRs, never main.** Phases that change code (E, F) work on a fresh
   `claude/<phase>-<date>` branch and open a PR.
4. **The digest is self-contained.** Session storage is ephemeral; any draft the
   maintainer will act on days later must appear **in full** in the digest text,
   not as a reference to a file in the container.
5. **Skip silently what has nothing to do.** A phase with no findings is one line
   in the digest ("no new issues"), not a section.

## Phase A — Community triage (every run)

Use the **maintainer-response** skill (`.claude/skills/maintainer-response/`).

- Window: everything since the last on-week run — **14 days** (overlap is fine;
  skip items already resolved).
- Collect: new issues, new PRs, new comments on open items, and replies to threads
  where the maintainer (JoshuaC215) was the last responder before the window.
- For each item needing a response, produce a draft reply per the skill's rules
  (research first, cite code, match scope to effort). Number the drafts in the
  digest so the maintainer can reply "post 1 and 3".
- Do NOT post any of these — the skill's draft-only rule holds.

## Phase B — Stale sweep (every run; the one pre-authorized write)

Criteria for **stale**: an open issue or PR where

- the last substantive activity (comment, commit push, review) is from
  **JoshuaC215**, and
- that activity is **60+ days old** (i.e. the other party never responded), and
- the item is not labeled `pinned` and the maintainer has not said to keep it open.

For each stale item: post a short, friendly closing comment — thank them, note
it's being closed for inactivity, and explicitly invite them to **re-open (or ask
for a re-open) if they want to continue** — then close the issue/PR
(`state_reason: not_planned` for issues). This is deliberately a one-step close
with no prior warning: re-opening is cheap and the invitation makes that clear.

List every close in the digest with a link.

## Phase C — Live app smoke test (every run)

```sh
uv run --with playwright python scripts/smoke_live_app.py
```

Drives https://agent-service-toolkit.streamlit.app/ in a real browser: wakes the
app if Streamlit Cloud put it to sleep, sends one message, verifies a response
streams back. Report pass/fail in the digest; on failure, attach the script's
log output and screenshot findings, and diagnose as far as read-only access
allows (is it the Streamlit front end, or the Azure-hosted agent service behind
it?).

## Phase D — Infra smoke tests (conditional)

Only if commits landed on `main` in the window that touch the paths listed in the
**smoke-test** skill's table (checkpointers/memory, AG-UI adapter, LangFuse
wiring, service lifespan, or bumps to those deps): run the narrowest relevant
`scripts/smoke_test.sh` targets. The SessionStart hook starts the Docker daemon;
if it isn't up, `(sudo dockerd >/tmp/dockerd.log 2>&1 &)` and wait a few seconds.
Otherwise skip with one digest line.

## Phase E — Model catalog refresh (first run of each month)

Use the **model-refresh** skill. Work on a fresh branch, open a PR summarizing
adds/removals/default changes with provider-doc citations. If provider API keys
are present in the environment, run `scripts/check_live_models.py` as the skill
directs; if not, note in the PR that live verification was skipped.

## Phase F — Dependency refresh (first run of each quarter)

Use the **dependency-refresh** skill (playbook: `docs/Dependency_Upgrades.md`).
Safe bumps in one PR; deferred majors recorded in the doc's backlog table with
ROI notes. Run the full verification ladder including the fake-model live e2e,
and the relevant infra smoke targets when checkpointer/AG-UI/LangFuse deps moved.

## Final step — The digest

End the session with **one** message, structured exactly as:

1. **Needs your decision** — numbered draft replies (full text, verbatim) and any
   flagged maintainer calls. This section first; it's why the human is reading.
2. **Done autonomously** — stale items closed (links), PRs opened (links).
3. **Health** — live app smoke result, infra smoke results (or "skipped: no
   relevant changes"), anything from CI worth knowing.
4. **Problems** — phases that failed or were skipped due to missing
   keys/allowlist/etc., each with a one-line cause.

If literally nothing happened in any phase (quiet fortnight, all green), say so
in three lines or fewer — that still gets delivered, since "all clear" is the
point of a scheduled report. The maintainer replies in-session to approve drafts
("post 1 and 3"); those follow-ups are ordinary maintainer-response skill
posting flows with explicit authorization.

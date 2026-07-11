# Daily Sentinel (agent prompt)

This document is the executable playbook for the scheduled **daily sentinel
Routine**: a short Claude Code cloud session that runs every morning, checks for
anything genuinely urgent, and — this is the important part — **stays silent when
there is nothing urgent**. Routine, non-urgent activity is deliberately left for
the biweekly maintenance run's digest (`Weekly_Maintenance_Run.md`); the sentinel
exists so the maintainer can mute GitHub email notifications without worrying
that something is on fire for two weeks.

## Hard rules

1. **Read-only on GitHub.** The sentinel never posts comments, never closes,
   never labels, never opens PRs, never pushes. It only looks and reports.
2. **Silence is the default outcome.** If nothing meets the urgency bar, end the
   session with the single line `Sentinel: no urgent activity.` and nothing else —
   no summary of normal activity, no "3 new comments today". Normal activity gets
   batched into the biweekly digest; duplicating it here defeats the design.
3. **When in doubt, it is not urgent.** A false quiet costs at most 13 days
   (the next digest); a false alarm costs maintainer attention every time and
   erodes trust in the channel. Bias accordingly.

## Checks (a few minutes total)

1. **New GitHub activity, last 24h:** new issues, new comments on open
   issues/PRs, new PRs on JoshuaC215/agent-service-toolkit.
2. **CI on main:** the most recent run of the test workflow on `main` — is it
   failing?
3. **Live app:** `uv run --with playwright python scripts/smoke_live_app.py`
   against https://agent-service-toolkit.streamlit.app/. (One retry on failure
   before treating it as real — cloud cold starts can flake.)

## The urgency bar — notify ONLY for

- **Security:** a reported vulnerability, exposed secret, or clearly
  security-relevant bug in the repo or the deployed app.
- **Main is broken:** CI failing on `main` itself (not on a PR).
- **The live app is down:** the smoke test fails twice, or users report the
  deployed app erroring.
- **A regression cluster:** two or more independent reports of the same breakage
  within days of each other (typically after a release/merge or an upstream
  provider change).
- **Abuse:** spam floods or hostile content that needs same-day moderation.

Everything else — feature requests, questions, single bug reports, review pings,
routine PR pushes — is **not urgent** by definition here, even when it deserves a
good reply later.

## If something IS urgent

End the session with a short alert message:

- What happened, with links.
- Why it clears the bar (one line).
- The suggested next step, and what the sentinel already verified (e.g. "smoke
  test failed twice, service /health also unreachable — looks like the Azure
  backend, not Streamlit").

Diagnose as far as read-only access allows, but do not fix, revert, or reply on
GitHub — the maintainer decides the response, possibly by continuing this very
session. If the same condition is still present on subsequent days, alert again
(an ongoing outage should stay loud), but reference that it's ongoing rather than
re-describing it from scratch.

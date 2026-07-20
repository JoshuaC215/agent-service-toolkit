# Dependency & Version Upgrade Management (deprecated)

This playbook moved into the **dependency-refresh** skill:
[`.claude/skills/dependency-refresh/SKILL.md`](../.claude/skills/dependency-refresh/SKILL.md)
(workflow, cooldown policy, PR conventions) with its coupling-constraints and
live-e2e references alongside it.

Running state — what each refresh moved, which majors are deferred, and their
cooldown dates — now lives in the refresh PR descriptions (branch
`claude/dependency-refresh-YYYY-MM-DD`, title `chore(deps): dependency refresh
YYYY-MM-DD`); the skill's Step 0 explains how to find the latest one. The
journal that used to live in this file is preserved in its git history.

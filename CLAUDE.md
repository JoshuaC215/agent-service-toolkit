# CLAUDE.md

## Comment style

- Default to no comments in code you add or edit.
- Only add a comment when the *why* is non-obvious: a hidden constraint, a subtle invariant, or a workaround for a specific bug. Never explain *what* the code does — clear naming should do that.
- Never write multi-paragraph comment blocks or over-explain inline. Match the terseness of the surrounding code.
- Exception: brand-new files may start with a short, genuinely useful module/file-level docstring. This does not license verbose inline comments throughout the rest of the file.
- When editing an existing file, match its existing comment density and style rather than introducing a heavier style than what's already there.

## Maintainer scaffolding vs. template content

This repo is a GitHub template. `.github/workflows/template-cleanup.yml` strips
maintainer-only scaffolding from downstream clones on first instantiation, driven by
conventions rather than a hardcoded list. Put new scaffolding where the cleanup already
covers it:

- Maintainer automation (skills, hooks, scheduled runs) goes in `.claude/` or
  `docs/maintenance/` — whole-directory buckets, removed entirely, auto-covered.
- A maintainer-only *workflow* must carry the owner-guard
  `if: github.repository == 'JoshuaC215/agent-service-toolkit'` line; the cleanup
  removes any guarded workflow, so this is auto-covered too.
- ONLY when a maintainer-only file must live in a shared dir (`scripts/`, etc.) and
  can't carry the guard marker do you add it to the explicit list in
  `template-cleanup.yml`. Adding such a file without updating the cleanup is the one
  case that leaks scaffolding into clones.

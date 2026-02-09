---
name: Git Operations
description: Git version control best practices including commits, branches, and merges
version: "1.0"
triggers:
  - type: semantic
    keywords: ["git", "commit", "branch", "merge", "push", "pull", "rebase", "version control"]
  - type: path_glob
    pattern: "**/.git/**"
resources:
  - scripts/commit_check.py
  - templates/commit_message.md
dependencies: []
---

# Git Operations Guide

## Commit Conventions

Use Conventional Commits format:

| Type | Description |
|------|-------------|
| `feat:` | New feature |
| `fix:` | Bug fix |
| `docs:` | Documentation update |
| `refactor:` | Code refactoring |
| `test:` | Test related |
| `chore:` | Build/tooling related |

### Example

```
feat(auth): add OAuth2 login support

- Add Google OAuth2 provider
- Add session management
- Update user model with oauth fields

Closes #123
```

## Branch Strategy

- `main`: Production branch, accepts merges only
- `develop`: Development branch
- `feature/*`: Feature branches, created from develop
- `hotfix/*`: Hotfix branches, created from main

## Merge Workflow

1. Ensure local branch is synced with remote: `git fetch origin`
2. Run tests to confirm they pass
3. Use `--no-ff` to preserve merge history: `git merge --no-ff feature/xxx`
4. Delete the merged feature branch

## Common Commands

```bash
# Check status
git status

# Stage changes
git add -p  # Interactive staging

# Commit
git commit -v  # Show diff

# Push
git push origin HEAD

# Rollback
git reset --soft HEAD~1  # Soft reset, keep changes
```

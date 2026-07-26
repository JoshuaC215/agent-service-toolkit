#!/bin/bash
# Pin the sandbox's uv to the repo's version. The relative `exclude-newer`
# cooldown in pyproject.toml needs a recent uv; an older uv silently drops the
# cooldown metadata from uv.lock on any re-lock (which `uv run`/`uv sync` can
# trigger). Installs from PyPI with the same `pip install uv==` method the
# Dockerfiles use, and derives the version from them so there's no extra copy to
# keep in sync. Only touches the managed remote sandbox -- never a local dev's
# uv install.
set -uo pipefail

[ "${CLAUDE_CODE_REMOTE:-}" = "true" ] || exit 0

project_dir="${CLAUDE_PROJECT_DIR:-.}"
pinned="$(grep -hoE 'uv==[0-9]+\.[0-9]+\.[0-9]+' "$project_dir"/docker/Dockerfile.* 2>/dev/null | head -1 | cut -d= -f3)"
[ -n "$pinned" ] || exit 0

current="$(uv --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)"
[ "$current" = "$pinned" ] && exit 0

python3 -m pip install --user --quiet --no-cache-dir "uv==$pinned" >/dev/null 2>&1 || true
exit 0

#!/usr/bin/env python3
"""Check if the last commit message follows Conventional Commits format."""

import re
import subprocess
import sys


def get_last_commit_message() -> str:
    result = subprocess.run(
        ["git", "log", "-1", "--pretty=%B"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def check_conventional_commit(message: str) -> bool:
    pattern = r"^(feat|fix|docs|refactor|test|chore|style|perf|ci|build|revert)(\(.+\))?: .+"
    return bool(re.match(pattern, message))


def main() -> None:
    message = get_last_commit_message()
    if not message:
        print("No commit message found.")
        sys.exit(1)

    if check_conventional_commit(message):
        print(f"✓ Commit message follows Conventional Commits: {message}")
    else:
        print(f"✗ Commit message does not follow Conventional Commits: {message}")
        print("  Expected format: type(scope): description")
        sys.exit(1)


if __name__ == "__main__":
    main()

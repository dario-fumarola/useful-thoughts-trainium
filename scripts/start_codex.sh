#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if ! command -v codex >/dev/null 2>&1; then
  printf 'Codex is not on PATH. Start your installed/authenticated Codex CLI in this folder and use CODEX_START_PROMPT.md.\n' >&2
  exit 2
fi
mkdir -p local
PROMPT="$(cat CODEX_START_PROMPT.md)"
if command -v flock >/dev/null 2>&1; then
  exec flock -n -E 75 local/codex-session.lock codex "$PROMPT"
else
  printf 'flock unavailable; ensure another coding session is not editing this worktree.\n' >&2
  exec codex "$PROMPT"
fi

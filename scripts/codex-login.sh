#!/usr/bin/env bash
set -euo pipefail
# docker exec … codex login --device-auth (headless)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CONTAINER="${CODEX_CONTAINER:-codex-to-api-codex}"
PATH_IN='/usr/local/bin:/usr/local/sbin:/usr/bin:/bin'

if [[ $# -gt 0 && ( $1 == status || $1 == help ) ]]; then
  exec docker exec -it -e "PATH=$PATH_IN" "$CONTAINER" codex login "$@"
fi
exec docker exec -it -e "PATH=$PATH_IN" "$CONTAINER" codex login --device-auth "$@"

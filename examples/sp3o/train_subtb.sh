#!/usr/bin/env bash
set -euo pipefail
export PRESET=subtb
bash "$(dirname -- "${BASH_SOURCE[0]}")/train.sh" "$@"

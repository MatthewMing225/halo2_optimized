#!/usr/bin/env bash
set -euo pipefail

crate_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)/ezkl"
cd "$crate_dir"

cargo check --features thread-safe-region,parallel-synthesis "$@"

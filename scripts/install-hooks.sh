#!/bin/sh
# Point git at the committed hooks directory. Idempotent; run once after clone.
_script_dir=$(cd "$(dirname "$0")" && pwd)
_repo_root=$(cd "$_script_dir/.." && pwd)
cd "$_repo_root"
git config core.hooksPath scripts/hooks
echo "install-hooks.sh: core.hooksPath = scripts/hooks"

#!/bin/sh
# scripts/envlock.sh -- was an environment's spack.lock resolved from its
# current spack.yaml?
#
# File times can't answer that: a fresh clone gives every file the same time,
# so a committed lock that predates a spack.yaml edit looks current. Instead,
# resolving records a checksum of spack.yaml in <dir>/spack.yaml.sha256
# (committed together with spack.lock), and the scripts compare against it.
#
#   scripts/envlock.sh check  DIR   exit 0: the lock was resolved from this spack.yaml
#                                   exit 1: spack.yaml changed since, or no lock
#                                   With no checksum recorded (an environment
#                                   resolved by hand or by an older checkout) it
#                                   falls back to file times: stale when
#                                   spack.yaml is newer than spack.lock.
#   scripts/envlock.sh record DIR   write DIR/spack.yaml.sha256; run right after
#                                   resolving (scripts/setup.sh does it itself)

cmd=${1:-}
dir=${2:-}
if [ -z "$dir" ] || { [ "$cmd" != check ] && [ "$cmd" != record ]; }; then
  echo "usage: envlock.sh check|record <environment directory>" >&2
  exit 64
fi
if [ ! -f "$dir/spack.yaml" ]; then
  echo "envlock.sh: no spack.yaml in $dir" >&2
  exit 64
fi

checksum()
{
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d' ' -f1
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | cut -d' ' -f1
  else
    python3 -c 'import hashlib, sys
print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$1"
  fi
}

if [ "$cmd" = record ]; then
  if [ ! -f "$dir/spack.lock" ]; then
    echo "envlock.sh: no spack.lock in $dir (resolve first)" >&2
    exit 1
  fi
  sum=$(checksum "$dir/spack.yaml") && [ -n "$sum" ] || exit 1
  echo "$sum" >"$dir/spack.yaml.sha256" || exit 1
  exit 0
fi

[ -f "$dir/spack.lock" ] || exit 1
if [ ! -s "$dir/spack.yaml.sha256" ]; then
  python3 -c 'import os, sys
sys.exit(1 if os.path.getmtime(sys.argv[1]) > os.path.getmtime(sys.argv[2]) + 2 else 0)' \
    "$dir/spack.yaml" "$dir/spack.lock"
  exit $?
fi
[ "$(checksum "$dir/spack.yaml")" = "$(sed -n '1{s/[[:space:]]//g;p;}' "$dir/spack.yaml.sha256")" ]

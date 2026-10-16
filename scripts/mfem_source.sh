#!/bin/sh
# scripts/mfem_source.sh [DIR] -- make the MFEM source tree the Spack
# environments build (their `develop:` section): MFEM v4.10 with mfem/mfem#5464
# (environments/patches/mfem-v4.10-pr5464.diff) committed on a local branch
# v4.10-pr5464. Needs internet (a login node). DIR defaults to
# $PROJECT/mfem-src/mfem-4.10, the path in environments/psc_*/spack.yaml.
#
# Why a patched v4.10: v4.10 contains PR #5297, whose
# ParMesh::AnisotropicConflict segfaults on nonconforming 3D hex meshes at
# np >= 2 (mfem/mfem#5602); the unmerged #5464 fixes it (docs/developing.md).
# Drop this script and the develop sections once #5464 is in a release.
set -e
repo=$(cd "$(dirname "$0")/.." && pwd)
patch="$repo/environments/patches/mfem-v4.10-pr5464.diff"
dir=${1:-${PROJECT:?set PROJECT or pass DIR}/mfem-src/mfem-4.10}
if [ -e "$dir" ]; then
  echo "mfem_source.sh: $dir exists; leaving it alone" >&2
  git -C "$dir" log --oneline -1 v4.10-pr5464 2>/dev/null || true
  exit 0
fi
git clone --depth 1 --branch v4.10 --single-branch https://github.com/mfem/mfem.git "$dir"
cd "$dir"
git switch -c v4.10-pr5464
git apply --index "$patch"
git -c user.name=bessemer -c user.email=bessemer@localhost commit -q \
  -m "Local patch: mfem/mfem#5464 (parncmesh-bugfix, its own diff vs its last master merge 3001edfcf4) on v4.10"
git log --oneline -2

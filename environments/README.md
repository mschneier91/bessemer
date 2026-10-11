# environments/

Spack environments, one per machine, plus the portable stack they're made from. How it
fits together and how to set up a machine: [docs/install/spack.md](../docs/install/spack.md).

| | |
|---|---|
| `stack.yaml` | what bessemer needs, for any machine; `scripts/setup.sh` builds a machine's environment from it |
| `desktop/` | the maintainer's workstation (reference lock) |
| `psc_gpu/` | PSC Bridges-2 H100 nodes, usable by any Bridges-2 user ([bridges2.md](../docs/install/bridges2.md)) |
| `psc_cpu/` | PSC Bridges-2, the same stack with no CUDA; selected with `INCNS_MACHINE=psc_cpu` ([bridges2.md](../docs/install/bridges2.md)) |
| `<name>/` | other machines, created by `scripts/setup.sh`; untracked unless the project maintains that machine |
| `<name>/spack.yaml.sha256` | checksum of the `spack.yaml` the lock was resolved from (`scripts/envlock.sh`); committed with the lock |
| `<name>/hosts` | optional: hostname patterns of the machines that use this environment (`scripts/machine.sh`) |
| `<name>/preset` | optional: the build preset this environment is for, when it isn't `cpu` (`scripts/setup.sh`, `scripts/doctor.sh`) |
| `<name>/INSTALL` | optional: names the document this environment installs by, when `scripts/setup.sh` can't do it |
| `.machine` | which environment this checkout uses (untracked; written by `setup.sh`) |

Edit a machine's `spack.yaml` only from a host terminal, and resolve and install right
after (traps 1 and 2 in docs/install/spack.md).

# Historical source snapshots

These 71 files retain the original implementation and experimental drivers from
`experiments/` and `gzc-impl/`. Their source hashes and original relative paths
are recorded in `../docs/source-inventory.json`.

**Use the public `couponim` commands to run new experiments.** This directory is
a reference archive, outside the installed Python package and CI test discovery.
The snapshots are not independently runnable as distributed: some depend on
original benchmark paths, external files, optional third-party libraries,
excluded drawing modules, or the original directory layout. They have not all
been revalidated. They are preserved so readers can inspect algorithm variants,
historical protocols, audits, resource supervision and acceleration code.

- `experiments/`: uniform and conditioned RR, streaming, forward baselines,
  parameter grids, adaptive selection, budget extensions, sampler/selection
  diagnosis, process scheduling and original tests.
- `gzc-impl/`: the earlier simulation, seed-selection, probability-distribution,
  sparse-matrix and synthetic-network utilities, plus its local configuration.
  Some functions implement fresh-decision/Markov or success-conditioned
  variants. Do not treat these as verified implementations of the current
  discard-on-revisit model without inspecting their semantics.

Original paths and dataset **names** in the source are historical configuration,
not bundled datasets. No graph files, measured results, mock results, images,
PDFs, compiled executables or plotting code are shipped. Duplicate source copies
inside old result directories are omitted. Conversion code using pickle remains
historical only; the supported loader accepts edge lists and numeric CSR NPZ.

This archive does not certify the conclusions of old experiment scripts.
Portable replacements and their supported behavior are listed in
`../docs/CATALOG.md`.

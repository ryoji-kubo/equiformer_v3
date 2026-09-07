# MPtrj and sAlex `max_atoms <= 20` Subsets

This note documents the change made to `experimental/datasets/mptrj_create_subset_aselmdb.py` and the commands to create small test subsets or full filtered subsets.

## What changed

The existing helper already created random ASE LMDB subsets. I extended it so it can also filter structures by atom count before sampling or writing.

New behavior:

- `--max-atoms 20` keeps only structures whose true number of atoms is `<= 20`.
- `--all-matching` writes every matching structure instead of randomly sampling `--num-samples`.
- The script prefers `metadata_num-nodes.npz` for atom counts, because in this repo `metadata.npz` is sometimes replaced with edge-count metadata for load balancing.
- If only `metadata.npz` exists, the script checks a few structures against the metadata values. If the values do not look like atom counts, it falls back to reading structures and using `len(atoms)`.
- The output dataset writes both `metadata.npz` and `metadata_num-nodes.npz` with true atom counts.
- The script refuses to append into an existing `data.aselmdb`, so reruns do not silently duplicate rows.
- The output ASE DB is closed in a `finally` block, so interrupts leave the process cleaner.

## Local source counts

From the local metadata on this machine:

- `dataset/mptrj/aselmdb_uncorrected_total_energy`: 1,578,762 total structures, 745,197 with `natoms <= 20`.
- `dataset/salex/val`: 553,218 total structures, 531,648 with `natoms <= 20`.

## Small test commands

Use these first to verify that the paths, environment, and output format work. These create only 1,000 samples from each filtered pool.

```bash
/home/ryoji/miniconda3/envs/equiformer_v3/bin/python \
  experimental/datasets/mptrj_create_subset_aselmdb.py \
  --source-dir dataset/mptrj/aselmdb_uncorrected_total_energy \
  --target-dir dataset/mptrj/aselmdb_uncorrected_total_energy_max_atoms_20_test_1k \
  --max-atoms 20 \
  --num-samples 1000 \
  --seed 0
```

```bash
/home/ryoji/miniconda3/envs/equiformer_v3/bin/python \
  experimental/datasets/mptrj_create_subset_aselmdb.py \
  --source-dir dataset/salex/val \
  --target-dir dataset/salex/val_max_atoms_20_test_1k \
  --max-atoms 20 \
  --num-samples 1000 \
  --seed 0
```

## Full filtered subset commands

After the small tests pass, use `--all-matching` to write every structure with at most 20 atoms.

```bash
/home/ryoji/miniconda3/envs/equiformer_v3/bin/python \
  experimental/datasets/mptrj_create_subset_aselmdb.py \
  --source-dir dataset/mptrj/aselmdb_uncorrected_total_energy \
  --target-dir dataset/mptrj/aselmdb_uncorrected_total_energy_max_atoms_20_full \
  --max-atoms 20 \
  --all-matching \
  --seed 0
```

```bash
/home/ryoji/miniconda3/envs/equiformer_v3/bin/python \
  experimental/datasets/mptrj_create_subset_aselmdb.py \
  --source-dir dataset/salex/val \
  --target-dir dataset/salex/val_max_atoms_20_full \
  --max-atoms 20 \
  --all-matching \
  --seed 0
```

## Quick validation

This checks that the output row count matches the metadata length and that the maximum atom count is no more than 20.

```bash
/home/ryoji/miniconda3/envs/equiformer_v3/bin/python - <<'PY'
from pathlib import Path
import numpy as np
from fairchem.core.datasets import AseDBDataset

for target in [
    Path('dataset/mptrj/aselmdb_uncorrected_total_energy_max_atoms_20_test_1k'),
    Path('dataset/salex/val_max_atoms_20_test_1k'),
]:
    ds = AseDBDataset({
        'src': str(target),
        'a2g_args': {'r_energy': True, 'r_forces': True, 'r_stress': True},
    })
    natoms = np.load(target / 'metadata_num-nodes.npz')['natoms']
    print(target)
    print('  rows:', len(ds))
    print('  metadata rows:', len(natoms))
    print('  max atoms:', int(natoms.max()))
PY
```

## Interrupted partial output

A full MPtrj run was started and then interrupted at about 264k rows. That means this directory is incomplete and should not be used as a finished dataset:

```text
dataset/mptrj/aselmdb_uncorrected_total_energy_max_atoms_20
```

Either remove that partial directory before reusing the same name, or use the `_full` target directory shown above.

import os
import argparse
from pathlib import Path

import ase
import numpy as np
import torch
from tqdm import tqdm

from fairchem.core.datasets import AseDBDataset


_NUM_SAMPLES = 10_000
_SOURCE_DIR = './aselmdb_uncorrected_total_energy'
_TARGET_DIR = './aselmdb_uncorrected_total_energy_10k'


def _load_natoms(source_dir, dataset, verify_samples=32):
    source_path = Path(source_dir)
    metadata_path = source_path / 'metadata_num-nodes.npz'
    if metadata_path.is_file():
        metadata = np.load(metadata_path)
        return metadata['natoms'], metadata_path

    metadata_path = source_path / 'metadata.npz'
    if not metadata_path.is_file():
        return None, None

    metadata = np.load(metadata_path)
    natoms = metadata['natoms']
    check_count = min(verify_samples, len(dataset))
    check_indices = np.linspace(0, len(dataset) - 1, check_count, dtype=int)
    for idx in check_indices:
        if int(natoms[idx]) != len(dataset.get_atoms(int(idx))):
            print(
                '{} does not appear to store atom counts; falling back to reading structures.'.format(
                    metadata_path
                )
            )
            return None, None
    return natoms, metadata_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a filtered random subset of an ASE LMDB dataset.')
    parser.add_argument('--num-samples', type=int, default=_NUM_SAMPLES, help='Number of samples for the subset.')
    parser.add_argument('--source-dir', type=str, default=_SOURCE_DIR, help='Source directory containing .aselmdb files.')
    parser.add_argument('--target-dir', type=str, default=_TARGET_DIR, help='Target directory for the subset.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility.')
    parser.add_argument('--max-atoms', type=int, default=None, help='Keep only structures with this many atoms or fewer.')
    parser.add_argument('--all-matching', action='store_true', help='Write every structure matching the filters instead of sampling.')
    args = parser.parse_args()

    torch.random.manual_seed(args.seed)

    dataset = AseDBDataset(
        {
            'src': args.source_dir,
            'a2g_args': {
                'r_energy': True,
                'r_forces': True,
                'r_stress': True,
            },
        }
    )
    length = len(dataset)
    print('Dataset length: {}'.format(length))

    natoms, metadata_path = _load_natoms(args.source_dir, dataset)
    if natoms is not None and len(natoms) != length:
        raise ValueError('Metadata length {} does not match dataset length {}'.format(len(natoms), length))
    if metadata_path is not None:
        print('Using atom-count metadata: {}'.format(metadata_path))

    indices = np.arange(length)
    if args.max_atoms is not None:
        if natoms is not None:
            indices = indices[natoms <= args.max_atoms]
        else:
            indices = np.array(
                [
                    i
                    for i in tqdm(indices, desc='Filtering by max atoms')
                    if len(dataset.get_atoms(int(i))) <= args.max_atoms
                ]
            )
        print('Structures with <= {} atoms: {}'.format(args.max_atoms, len(indices)))

    if args.all_matching:
        idx_list = indices
    else:
        if args.num_samples > len(indices):
            raise ValueError(
                'Requested {} samples but only {} structures are available'.format(
                    args.num_samples, len(indices)
                )
            )
        idx_list = indices[torch.randperm(len(indices)).numpy()[: args.num_samples]]

    os.makedirs(args.target_dir, exist_ok=True)
    output_path = os.path.join(args.target_dir, 'data.aselmdb')
    if os.path.exists(output_path):
        raise FileExistsError('{} already exists'.format(output_path))

    db = ase.db.connect(output_path)
    natoms_list = []
    try:
        for idx in tqdm(idx_list, desc='Writing subset'):
            atoms = dataset.get_atoms(int(idx))
            db.write(atoms, data=atoms.info)
            natoms_list.append(len(atoms))
    finally:
        if hasattr(db, 'close'):
            db.close()

    natoms_array = np.array(natoms_list)
    np.savez(os.path.join(args.target_dir, 'metadata.npz'), natoms=natoms_array)
    np.savez(os.path.join(args.target_dir, 'metadata_num-nodes.npz'), natoms=natoms_array)
    print('Wrote {} structures to {}'.format(len(natoms_array), args.target_dir))

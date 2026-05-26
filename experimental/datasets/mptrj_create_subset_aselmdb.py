import os
import argparse
import torch
import ase
from tqdm import tqdm
import numpy as np
from fairchem.core.datasets import AseDBDataset


_NUM_SAMPLES = 10_000
_SOURCE_DIR = './aselmdb_uncorrected_total_energy'
_TARGET_DIR = './aselmdb_uncorrected_total_energy_10k'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create a random subset of an ASE LMDB dataset.")
    parser.add_argument("--num-samples", type=int, default=_NUM_SAMPLES, help="Number of samples for the subset.")
    parser.add_argument("--source-dir", type=str, default=_SOURCE_DIR, help="Source directory containing .aselmdb files.")
    parser.add_argument("--target-dir", type=str, default=_TARGET_DIR, help="Target directory for the subset.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    args = parser.parse_args()

    torch.random.manual_seed(args.seed)
    
    dataset = AseDBDataset(
        {
            'src': args.source_dir,
            'a2g_args': {
                'r_energy': True,
                'r_forces': True,
                'r_stress': True,
            }
        }
    )
    length = len(dataset)
    print('Dataset length: {}'.format(length))
    idx_list = torch.randperm(length)

    os.makedirs(args.target_dir, exist_ok=True)
    db = ase.db.connect(os.path.join(args.target_dir, 'data.aselmdb'))
    natoms_list = []
    for i in tqdm(range(args.num_samples)):
        atoms = dataset.get_atoms(idx_list[i])
        db.write(atoms, data=atoms.info)
        natoms_list.append(len(atoms))
    np.savez(
        os.path.join(args.target_dir, 'metadata.npz'),
        natoms=(np.array(natoms_list))
    )
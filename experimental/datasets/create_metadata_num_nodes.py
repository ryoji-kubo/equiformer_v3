import os
import numpy as np
from tqdm import tqdm
import typer
from typing import Annotated

from fairchem.core.datasets import AseDBDataset


def create_metadata(
    input_dir: Annotated[
        str, typer.Option(help="Input directory to .aselmdb files")
    ],
) -> None:

    dataset = AseDBDataset(
        {
            'src': input_dir,
            'a2g_args': {
                'r_energy': True,
                'r_forces': True,
                'r_stress': True,
            }
        }
    )
    length = len(dataset)
    print('Dataset length: {}'.format(length))
    
    natoms_list = []
    for i in tqdm(range(len(dataset))):
        atoms = dataset.get_atoms(i)
        natoms_list.append(len(atoms))
        
    np.savez(
        os.path.join(input_dir, 'metadata_num-nodes.npz'),
        natoms=(np.array(natoms_list))
    )


if __name__ == '__main__':
    typer.run(create_metadata)
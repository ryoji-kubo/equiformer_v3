import torch
from torch_geometric.utils import to_dense_batch

from .coordinate_utils import cartesian_to_fractional_dense

def debug_compare_fractional_coords_with_pymatgen(
    data,
    wrap: bool = True,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> dict[str, object]:
    """
    Debug helper to verify the transformer's Cartesian-to-fractional conversion
    against pymatgen.core.Lattice.get_fractional_coords.

    This is intentionally not used in training or evaluation. It imports
    pymatgen lazily so transformer inference does not depend on pymatgen.
    """
    try:
        from pymatgen.core import Lattice
    except ImportError as exc:
        raise ImportError(
            "Install pymatgen to use debug_compare_fractional_coords_with_pymatgen."
        ) from exc

    import numpy as np

    batch = getattr(data, "batch", None)
    if batch is None:
        batch = torch.zeros_like(data.atomic_numbers.long())
    else:
        batch = batch.long()

    pos_dense, atom_mask = to_dense_batch(data.pos, batch)
    batch_size = pos_dense.shape[0]
    cell = data.cell
    if cell.dim() == 2 and cell.shape == (3, 3):
        cell = cell.unsqueeze(0)
    cell = cell.reshape(batch_size, 3, 3)

    torch_frac = cartesian_to_fractional_dense(pos_dense, cell, wrap=wrap)
    torch_frac_valid = torch_frac[atom_mask]

    pos_np = pos_dense.detach().cpu().double().numpy()
    cell_np = cell.detach().cpu().double().numpy()
    atom_mask_np = atom_mask.detach().cpu().numpy()

    pymatgen_frac_list = []
    for idx in range(batch_size):
        frac = Lattice(cell_np[idx]).get_fractional_coords(pos_np[idx, atom_mask_np[idx]])
        if wrap:
            frac = np.remainder(frac, 1.0)
        pymatgen_frac_list.append(frac)

    pymatgen_frac = torch.as_tensor(
        np.concatenate(pymatgen_frac_list, axis=0),
        dtype=torch_frac_valid.dtype,
        device=torch_frac_valid.device,
    )

    diff = torch_frac_valid - pymatgen_frac
    if wrap:
        comparison_diff = torch.remainder(diff + 0.5, 1.0) - 0.5
    else:
        comparison_diff = diff
    comparison_abs_diff = comparison_diff.abs()
    tolerance = atol + rtol * pymatgen_frac.abs()

    return {
        "batch_size": batch_size,
        "num_atoms": int(atom_mask.sum().item()),
        "wrap": wrap,
        "max_abs_diff": float(comparison_abs_diff.max().item()),
        "mean_abs_diff": float(comparison_abs_diff.mean().item()),
        "allclose": bool(torch.all(comparison_abs_diff <= tolerance).item()),
    }
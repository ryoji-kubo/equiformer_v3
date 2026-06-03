# FAIRChem / Open Catalyst Project DataBatch Attributes

This document provides a comprehensive explanation of the attributes found in a typical PyTorch Geometric `DataBatch` object used within the FAIRChem / Open Catalyst Project (OCP) framework, specifically tailored for trainers like the `EquiformerV3DeNSTrainer`.

As a running example, we will dissect the following batched object representing **8 individual structures** and a total of **203 atoms**:
`DataBatch(pos=[203, 3], cell=[8, 3, 3], atomic_numbers=[203], natoms=[8], tags=[203], sid=[8], energy=[8], forces=[203, 3], stress=[24, 3], fixed=[203], pbc=[24], fid=[8], batch=[203], ptr=[9], noise_mask=[203], noise_vec=[203, 3], denoising_pos_forward=True, dens_batch_mask=[8])`

---

## 1. Atom-Level Attributes (Length 203)
These tensors contain features mapping to every individual atom across all structures in the batch.

* **`atomic_numbers=[203]`**: The atomic number (Z) of each atom. Used as an **explicit input** to generate the initial continuous node embeddings (e.g., passing through `nn.Embedding`).
* **`pos=[203, 3]`**: The 3D Cartesian coordinates (x, y, z) of each atom. Used as an **explicit input** to compute interatomic vectors, distances, and spherical harmonics for message passing.
* **`tags=[203]`**: Classification tags for atoms: `0` (bulk/sub-surface), `1` (surface), and `2` (adsorbate). While present in the dataset, Equiformer V3 generally ignores this attribute, whereas older models (like GemNet) used it explicitly to restrict calculations (like quadruplet interactions) to specific atom types.
* **`fixed=[203]`**: A boolean/integer mask (`1` for fixed, `0` for free) indicating whether an atom's position is frozen during structural relaxation. It is **implicitly used** during training to mask out the loss and evaluation metrics for frozen sub-surface atoms (so the model isn't penalized for force predictions on atoms that cannot move).
* **`forces=[203, 3]`**: The ground-truth 3D force vectors (fx, fy, fz) acting on each atom. This is an **atom-level target** the model is trained to predict.

## 2. System-Level Attributes (Length 8)
These tensors describe global properties of the 8 separate crystal/molecular systems inside the batch.

* **`natoms=[8]`**: The number of atoms belonging to each individual structure. The sum of this array equals 203.
* **`cell=[8, 3, 3]`**: The 3x3 periodic bounding box vectors defining the unit cell dimensions for each structure. **Explicitly used** alongside positions and PBCs to build the periodic neighbor graph.
* **`pbc=[24]`**: Periodic Boundary Conditions. Typically an `[8, 3]` boolean array flattened to length 24. Indicates whether periodic boundaries are applied along the x, y, and z directions of the unit cell.
* **`energy=[8]`**: The total potential energy of the system. This is the **system-level target** the model is trained to predict.
* **`stress=[24, 3]`**: The stress tensor of the unit cell, typically an `[8, 3, 3]` tensor flattened to `[24, 3]`. Used as a **target** for bulk/cell relaxation tasks.
* **`sid=[8]`**: System ID. **Metadata** representing a unique identifier (string or integer) mapping the structure back to the source dataset.
* **`fid=[8]`**: Frame ID. **Metadata** tracking the specific time step (frame) of the structure if it originates from a relaxation trajectory.

## 3. PyTorch Geometric Batching Attributes
Because PyG processes batches as a single giant disconnected graph, it relies on indexing pointers to segment the atoms back to their respective graphs.

* **`batch=[203]`**: An assignment vector mapping each of the 203 atoms to its parent system index (ranging from `0` to `7`). Used heavily in scatter/gather pooling operations (e.g., summing atomic energies into system energies).
* **`ptr=[9]`**: A pointer vector indicating the exact array start and end indices of the atoms for each system. It begins with `0` and ends with the total number of atoms (`203`).

## 4. DeNS (Denoising Non-Equilibrium Structures) Attributes
These attributes are created on the fly by the `EquiformerV3DeNSTrainer`. They are specific to the auxiliary denoising task, which artificially corrupts atomic coordinates to improve model robustness.

* **`noise_vec=[203, 3]`**: The 3D Gaussian noise vector that was added to the `pos` coordinates. During the denoising forward pass, predicting this vector becomes the surrogate **atom-level target** instead of `forces`.
* **`noise_mask=[203]`**: A boolean mask indicating which specific atoms had noise added to them. This is used when `corrupt_ratio` < 1.0 is specified in the config, meaning only a fraction of the free atoms are perturbed.
* **`dens_batch_mask=[8]`**: A system-level boolean mask indicating which of the 8 structures in the batch actively had the DeNS task applied. Structures might be skipped (masked out) if they have too few atoms, fall under specific filter criteria (`max_force_norm`, `strict_max_ratio`, etc.), or have a manual `skip_dens` flag set in the database.
* **`denoising_pos_forward=True`**: A boolean flag telling the trainer's logic that the current iteration is operating in "denoising mode". It routes the loss calculation to compute the error against `noise_vec` instead of the standard DFT `forces`.

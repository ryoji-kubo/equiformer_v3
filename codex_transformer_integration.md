# Transformer Integration Summary

This note summarizes the transformer integration added for direct energy and
force prediction in `equiformer_v3`.

## Goal

The original transformer architecture in
`../Material-Equivariance/ryoji_1mar26/125_fm_llama_mp20_vsun_xb_v1.ipynb`
was a flow-matching model. For this integration, the architecture was adapted
into a FairChem-style direct predictor that consumes a structure and returns
the same output dictionary expected by
`experimental/trainers/equiformer_v3_dens_trainer.py`.

The direct forward path is the only supported path for now. DeNS and
gradient-based force/stress prediction were intentionally left out of scope.

## Files Added

- `experimental/models/transformer/transformer.py`
  - Defines the registered model `transformer`.
  - Implements a LLaMA-style transformer using RMSNorm, content-only self-attention, and
    SwiGLU blocks.
  - Converts PyG batches into padded atom sequences using `to_dense_batch`.
  - Uses atoms as tokens, with embeddings from `data.atomic_numbers` and
    encoded Cartesian positions from `data.pos`.
  - Encodes `data.cell` as both a per-structure cell token and a per-layer
    conditioning vector.
  - Produces `energy` and `stress` at system level and `forces` at atom level.

- `experimental/configs/omat24/mptrj/experiments/direct/transformer_160k_NoDeNS.yml`
  - A first runnable direct-training config using
    `equiformer_v3_dens_trainer`.
  - Sets `model.name: transformer`.
  - Trains/evaluates `energy`, `forces`, and `stress`.
  - Sets `use_denoising_pos: False`.
  - Sets `use_compile: False` for the first integration pass.

## Model Contract

The model is registered with:

```yaml
model:
  name: transformer
```

The trainer builds it through the existing FairChem registry:

```python
registry.get_model_class(model_name)(**model_config)
```

The model forward takes a batched PyG `data` object and returns:

```python
{
    "energy": Tensor[num_structures],
    "forces": Tensor[num_atoms, 3],
    "stress": Tensor[num_structures, 9],
}
```

This matches the direct output contract used by the existing Equiformer V3
models in the DeNS trainer.

## Architecture Notes

The notebook's transformer design was adapted as follows:

- Flow-matching time embeddings and velocity decoders were removed.
- Atom-type one-hot inputs were replaced with an `nn.Embedding` over
  `data.atomic_numbers`.
- Position inputs come from `data.pos`.
- Cell inputs come from `data.cell.reshape(batch_size, 9)`.
- A cell token is prepended to the atom sequence.
- The cell embedding is also added as a per-layer sequence condition.
- Atom features are decoded into atomwise force vectors.
- Atom features are decoded into atomwise energy contributions, summed per
  structure, and optionally combined with a cell-token energy head.
- Atom features are decoded into atomwise stress contributions, averaged per
  structure, and optionally combined with a cell-token stress head.

## Forward Integration

`StructureTransformer.forward()` is the main compatibility layer between the atom-token transformer and the existing FairChem trainer contract.

The incoming `data` object is a PyG batch where atom fields are flattened across all structures. The forward pass first calls `_dense_inputs(data)`, which:

- Reads `data.atomic_numbers`, `data.pos`, `data.batch`, and `data.cell`.
- Uses `to_dense_batch` to convert flattened atom positions into `pos_dense` with shape `[batch_size, max_atoms, 3]`.
- Uses `to_dense_batch` again to convert atomic numbers into `atomic_numbers_dense` with shape `[batch_size, max_atoms]`.
- Builds `atom_mask`, which marks real atoms and excludes padding.
- Optionally centers positions in each structure around its atomwise mean.
- Flattens `data.cell` into `[batch_size, 9]`.

After that, atom tokens are built by adding the encoded position and atom-type embedding:

```python
atom_tokens = (
    self.position_encoder(pos_dense)
    + self.atom_embedding(atomic_numbers_dense)
)
```

Padding tokens are zeroed with `atom_mask`, then a cell token is prepended:

```python
cell_token = self.cell_token_encoder(cell_flat).unsqueeze(1)
tokens = torch.cat((cell_token, atom_tokens), dim=1)
```

The same cell information is also encoded as `cell_condition` and added before each transformer block. This is the per-sequence conditioning path for `data.cell`:

```python
for block in self.blocks:
    tokens = tokens + cell_condition
    tokens = block(tokens, valid_mask)
```

`valid_mask` includes the cell token plus real atom tokens, so attention masks out padded atoms. No atom-index positional encoding is used, which keeps the attention stack permutation equivariant over atom tokens.

After the transformer stack, the output is split back into the cell feature and atom features:

```python
cell_features = tokens[:, 0, :]
atom_features = tokens[:, 1 : max_atoms + 1, :]
```

Energy is predicted from atom features as per-atom contributions, masked to remove padding, summed per structure, and optionally augmented with the cell-token energy head:

```python
atom_energy = self.atom_energy_head(atom_features).squeeze(-1)
atom_energy = atom_energy * atom_mask.to(dtype=atom_energy.dtype)
energy = atom_energy.sum(dim=1) / self.avg_num_nodes
```

Forces are predicted from atom features and then unpadded with `atom_mask`:

```python
outputs["forces"] = self.force_head(atom_features)[atom_mask]
```

This returns force predictions in the original flattened atom order, with shape `[num_atoms, 3]`, matching what `equiformer_v3_dens_trainer.py` expects for an atom-level `forces` target. The returned energy has shape `[batch_size]`, matching the system-level `energy` target.

Stress is predicted as a system-level 9-component tensor. The model decodes atomwise stress contributions from atom features, masks padding, averages over real atoms, and optionally adds a cell-token stress correction:

```python
atom_stress = self.atom_stress_head(atom_features)
stress_mask = atom_mask.unsqueeze(-1).to(dtype=atom_stress.dtype)
num_atoms = stress_mask.sum(dim=1).clamp_min(1.0)
stress = (atom_stress * stress_mask).sum(dim=1) / num_atoms
if self.include_cell_stress:
    stress = stress + self.cell_stress_head(cell_features)
outputs["stress"] = stress
```

This returns `stress` with shape `[batch_size, 9]`, matching the trainer system-level stress target.

## Config Usage

Use the new config as a starting point:

```bash
cd /home/ryoji/equivarient/equiformer_v3
/home/ryoji/miniconda3/bin/conda run -n mat_eq_v2 python my_main.py \
  --config-yml experimental/configs/omat24/mptrj/experiments/direct/transformer_160k_NoDeNS.yml
```

Make sure `equiformer_v3/src` is ahead of
`../Material-Equivariance/src` in `PYTHONPATH` if both repositories are on the
environment path. During verification, importing FairChem from
`Material-Equivariance` first caused the registry not to see the new
`transformer` model.

## Verification Performed

The following checks passed:

- Python compile check for `experimental/models/transformer/transformer.py`.
- Registry import check with `equiformer_v3/src` first on `sys.path`.
- Instantiation from the YAML model block:
  - `StructureTransformer`
  - `49,734,167` parameters
- Full-size synthetic forward pass:
  - `energy.shape == torch.Size([2])`
  - `forces.shape == torch.Size([5, 3])`
  - `stress.shape == torch.Size([2, 9])`
- Atom permutation smoke test with stress enabled:
  - energy max diff: `2.38e-7`
  - force max diff after unpermuting: `8.94e-8`
  - stress max diff: `7.45e-8`

## Current Limitations

- Only direct prediction is supported.
- `direct_prediction: False` raises an error in the transformer model.
- DeNS-specific behavior is disabled in the provided config.
- Stress prediction is supported in the provided config, but it is a direct MLP head rather than Equiformer V3 SO(3)-structured stress head.
- This transformer is not equivariant; it is a sequence model over atom tokens
  with position and cell features.

## Git Note

The original `.gitignore` rule `models/` ignored
`experimental/models/transformer/transformer.py` because it matched any
directory named `models`. Changing that ignore rule to target logs/checkpoints
instead makes the transformer source visible to Git.

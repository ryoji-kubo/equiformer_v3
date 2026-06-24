`cartesian_to_fractional_dense()` from `coordinate_utils.py` converts Cartesian atom coordinates into fractional coordinates for a dense batch.

Shapes:

```python
pos_dense: [B, N, 3]
cell:      [B, 3, 3]
```

where:

- `B` = batch size
- `N` = max atoms after padding
- `pos_dense[b, i]` = Cartesian xyz coordinate of atom `i` in structure `b`
- `cell[b]` = 3 lattice vectors for structure `b`

The geometry relation is:

```text
cartesian = fractional @ cell
```

So to recover fractional coordinates:

```text
fractional = cartesian @ inverse(cell)
```

Instead of explicitly computing `inverse(cell)`, the code solves a linear system, which is numerically better:

```python
frac_pos = torch.linalg.solve(
    cell.transpose(1, 2),
    pos_dense.transpose(1, 2),
).transpose(1, 2)
```

Why the transposes? `torch.linalg.solve(A, B)` solves:

```text
A @ X = B
```

For each batch item, we want:

```text
frac @ cell = pos
```

Transpose both sides:

```text
cell.T @ frac.T = pos.T
```

So the code solves:

```python
cell.T @ frac.T = pos.T
```

then transposes back to get `[B, N, 3]`.

Then:

```python
frac_pos = torch.remainder(frac_pos, 1.0)
```

wraps fractional coordinates into `[0, 1)`. For example:

```text
1.2  -> 0.2
-0.1 -> 0.9
```

That means atoms are represented inside the canonical unit cell.

So the function returns fractional coordinates with the same batch/atom shape as the input:

```python
[B, N, 3]
```

and, if `wrap=True`, values should be in `[0, 1)` up to numerical precision.
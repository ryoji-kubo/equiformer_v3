from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.utils import to_dense_batch

from fairchem.core.common.registry import registry


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


def _build_mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    dropout: float,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.SiLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.shape
        q = self.q_proj(x).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        k = self.k_proj(x).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        v = self.v_proj(x).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = torch.matmul(q.float(), k.float().transpose(-2, -1)) * self.scale
        attn = attn.masked_fill(
            ~valid_mask.view(batch_size, 1, 1, seq_len),
            torch.finfo(attn.dtype).min,
        )
        attn = F.softmax(attn, dim=-1).to(v.dtype)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        return self.out_proj(out)


class SwiGLU(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(2.0 * embed_dim / 3.0)
        self.gate_proj = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attn_dropout: float,
        residual_dropout: float,
        mlp_hidden_dim: Optional[int],
        mlp_dropout: float,
    ):
        super().__init__()
        self.attn_norm = RMSNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, attn_dropout)
        self.attn_drop = nn.Dropout(residual_dropout)
        self.mlp_norm = RMSNorm(embed_dim)
        self.mlp = SwiGLU(embed_dim, mlp_hidden_dim)
        self.mlp_drop = nn.Dropout(mlp_dropout)

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attn_drop(self.attn(self.attn_norm(x), valid_mask))
        x = x + self.mlp_drop(self.mlp(self.mlp_norm(x)))
        return x * valid_mask.unsqueeze(-1).to(dtype=x.dtype)


@registry.register_model("transformer")
class StructureTransformer(nn.Module):
    """
    LLaMA-style direct predictor for structures.

    Atoms are sequence tokens. Each token receives an atomic-number embedding and
    an encoded Cartesian position. The cell is encoded as a per-structure token
    and as a per-layer sequence condition.
    """

    def __init__(
        self,
        max_num_elements: int = 128,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        encoder_hidden_dim: Optional[int] = None,
        mlp_hidden_dim: Optional[int] = None,
        dropout: float = 0.05,
        attn_dropout: Optional[float] = None,
        residual_dropout: Optional[float] = None,
        mlp_dropout: Optional[float] = None,
        position_scale: float = 1.0,
        center_positions: bool = True,
        include_cell_energy: bool = True,
        include_cell_stress: bool = True,
        regress_forces: bool = True,
        regress_stress: bool = False,
        direct_prediction: bool = True,
        avg_num_nodes: float = 1.0,
        d: Optional[int] = None,
    ):
        super().__init__()
        if d is not None:
            embed_dim = d
        if not direct_prediction:
            raise ValueError("StructureTransformer currently supports direct_prediction only")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.max_num_elements = max_num_elements
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.position_scale = position_scale
        self.center_positions = center_positions
        self.include_cell_energy = include_cell_energy
        self.include_cell_stress = include_cell_stress
        self.regress_forces = regress_forces
        self.regress_stress = regress_stress
        self.avg_num_nodes = avg_num_nodes

        if encoder_hidden_dim is None:
            encoder_hidden_dim = int(1.5 * embed_dim)
        if attn_dropout is None:
            attn_dropout = dropout
        if residual_dropout is None:
            residual_dropout = dropout
        if mlp_dropout is None:
            mlp_dropout = dropout

        self.atom_embedding = nn.Embedding(
            max_num_elements,
            embed_dim,
            padding_idx=0,
        )
        self.position_encoder = _build_mlp(3, encoder_hidden_dim, embed_dim, dropout)
        self.cell_token_encoder = _build_mlp(9, encoder_hidden_dim, embed_dim, dropout)
        self.cell_condition_encoder = _build_mlp(
            9,
            encoder_hidden_dim,
            embed_dim,
            dropout,
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    attn_dropout=attn_dropout,
                    residual_dropout=residual_dropout,
                    mlp_hidden_dim=mlp_hidden_dim,
                    mlp_dropout=mlp_dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = RMSNorm(embed_dim)

        self.atom_energy_head = _build_mlp(embed_dim, encoder_hidden_dim, 1, dropout)
        self.cell_energy_head = _build_mlp(embed_dim, encoder_hidden_dim, 1, dropout)
        self.force_head = _build_mlp(embed_dim, encoder_hidden_dim, 3, dropout)
        self.atom_stress_head = _build_mlp(embed_dim, encoder_hidden_dim, 9, dropout)
        self.cell_stress_head = _build_mlp(embed_dim, encoder_hidden_dim, 9, dropout)

    def _dense_inputs(
        self,
        data,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        atomic_numbers = data.atomic_numbers.long()
        if atomic_numbers.numel() > 0 and atomic_numbers.min() <= 0:
            raise ValueError(
                "StructureTransformer expects atomic_numbers to be real elements >= 1; "
                "0 is reserved for padding."
            )
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros_like(atomic_numbers)
        else:
            batch = batch.long()
        
        # change it to dense representation (batch_size, max_num_atoms, feature_dim)
        pos_dense, atom_mask = to_dense_batch(data.pos, batch)
        atomic_numbers_dense, _ = to_dense_batch(atomic_numbers, batch)

        if self.center_positions:
            mask = atom_mask.unsqueeze(-1).to(dtype=pos_dense.dtype)
            count = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            center = (pos_dense * mask).sum(dim=1, keepdim=True) / count
            pos_dense = (pos_dense - center) * mask

        pos_dense = pos_dense / self.position_scale

        batch_size = pos_dense.shape[0]
        cell = data.cell
        if cell.dim() == 2 and cell.shape == (3, 3):
            cell = cell.unsqueeze(0)
        cell = cell.reshape(batch_size, 3, 3)
        cell_flat = cell.reshape(batch_size, 9) / self.position_scale

        return pos_dense, atomic_numbers_dense, atom_mask, cell_flat

    def forward(self, data) -> dict[str, torch.Tensor]:
        pos_dense, atomic_numbers_dense, atom_mask, cell_flat = self._dense_inputs(data)
        batch_size, max_atoms, _ = pos_dense.shape

        atom_tokens = (
            self.position_encoder(pos_dense)
            + self.atom_embedding(atomic_numbers_dense)
        )
        atom_tokens = atom_tokens * atom_mask.unsqueeze(-1).to(dtype=atom_tokens.dtype)

        cell_token = self.cell_token_encoder(cell_flat).unsqueeze(1)
        tokens = torch.cat((cell_token, atom_tokens), dim=1)

        cell_mask = torch.ones(
            batch_size,
            1,
            device=atom_mask.device,
            dtype=torch.bool,
        )
        valid_mask = torch.cat((cell_mask, atom_mask), dim=1)
        cell_condition = self.cell_condition_encoder(cell_flat).unsqueeze(1)

        for block in self.blocks:
            tokens = tokens + cell_condition
            tokens = block(tokens, valid_mask)

        tokens = self.norm(tokens)
        cell_features = tokens[:, 0, :]
        atom_features = tokens[:, 1 : max_atoms + 1, :]

        atom_energy = self.atom_energy_head(atom_features).squeeze(-1)
        atom_energy = atom_energy * atom_mask.to(dtype=atom_energy.dtype)
        energy = atom_energy.sum(dim=1) / self.avg_num_nodes
        if self.include_cell_energy:
            energy = energy + self.cell_energy_head(cell_features).squeeze(-1)

        outputs = {"energy": energy}
        if self.regress_forces:
            outputs["forces"] = self.force_head(atom_features)[atom_mask]
        if self.regress_stress:
            atom_stress = self.atom_stress_head(atom_features)
            stress_mask = atom_mask.unsqueeze(-1).to(dtype=atom_stress.dtype)
            num_atoms = stress_mask.sum(dim=1).clamp_min(1.0)
            stress = (atom_stress * stress_mask).sum(dim=1) / num_atoms
            if self.include_cell_stress:
                stress = stress + self.cell_stress_head(cell_features)
            outputs["stress"] = stress
        return outputs

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @torch.jit.ignore
    def no_weight_decay(self) -> set[str]:
        no_wd = set()
        for name, module in self.named_modules():
            if isinstance(module, (nn.Embedding, RMSNorm, nn.LayerNorm)):
                for parameter_name, _ in module.named_parameters(recurse=False):
                    no_wd.add(f"{name}.{parameter_name}" if name else parameter_name)

        for name, _ in self.named_parameters():
            if name.endswith("bias"):
                no_wd.add(name)
        return no_wd

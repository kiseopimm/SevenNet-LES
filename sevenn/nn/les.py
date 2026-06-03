"""
LES (Latent Ewald Summation) modules for SevenNet.

Architecture:
  NODE_FEATURE (last conv. layer, all-scalar)
      │
      ├─→ [LatentChargeReadout] → LES_Q (N_atoms, n_charges)
      │
      └─→ [init_feature_reduce] → ATOMIC_ENERGY → [AtomReduce] → SR_ENERGY
                                                                       │
  LES_Q ──→ [LatentEwaldSum] ──→ LR_ENERGY ──→ [AddLREnergy] ─────────┘
                                                       │
                                              PRED_TOTAL_ENERGY
                                                       │
                                           [LESForceStressOutput]

Forces:
  F_edge   = -d(E_total)/d(EDGE_VEC)  SR forces + q-path LR forces
  F_lr_pos = -d(E_LR)/d(POS)          direct Ewald positional forces

Stress:
  σ_edge = -(1/V) Σ_ij F_ij ⊗ r_ij   edge virial (SR + q-path LR)
  σ_lr   = -(1/V) d(E_LR)/d(LES_STRAIN)  complete Ewald stress via
             affine strain applied to pos and cell before les()

EDGE_VEC and POS are independent precomputed leaves, so the three
gradient paths are non-overlapping.

References:
  - LES library: https://github.com/ChengUCB/les
  - NequIP-LES:  https://github.com/ChengUCB/nequip-les
"""
from typing import Optional

import torch
import torch.nn as nn
from e3nn.o3 import Irreps

import sevenn._keys as KEY
from sevenn._const import AtomGraphDataType

from .linear import IrrepsLinear
from .util import broadcast


class LatentChargeReadout(nn.Module):
    """
    Projects node features to per-atom latent charges.

    Architecture (controlled by ``hidden_channels``):
        hidden_channels=[] (default):
            irreps_in ──[IrrepsLinear]──► (N, n_charges)
        hidden_channels=[H, ...]:
            irreps_in ──[IrrepsLinear]──► (N, H) ──[SiLU + nn.Linear]──► (N, n_charges)

    The first layer is SevenNet's IrrepsLinear (a thin wrapper around
    e3nn.o3.Linear that operates on AtomGraphData dicts). Modality dependence
    flows in through the upstream conv stack's modality-aware features; this
    layer does not concatenate the modality one-hot itself.

    Args:
        irreps_in:       e3nn irreps of the input node features.
        n_charges:       number of latent charge channels per atom (default 1).
                         With n_charges > 1 the Ewald energy is the sum of
                         n_charges independent Coulomb interactions, one per
                         channel: E_LR = Σ_α E_Coulomb(q^α).
        hidden_channels: hidden layer widths, e.g. [128] for one hidden layer.
        zero_init:       zero-initialise all weights so E_LR = 0 at init.
                         Useful for transparent-wrapper tests; not for training.
    """

    def __init__(
        self,
        irreps_in: Irreps,
        data_key_in: str = KEY.NODE_FEATURE,
        data_key_out: str = KEY.LES_Q,
        n_charges: int = 1,
        hidden_channels: Optional[list] = None,
        zero_init: bool = False,
    ):
        super().__init__()
        self.key_input = data_key_in
        self.key_output = data_key_out
        self.n_charges = n_charges

        if hidden_channels is None:
            hidden_channels = []
        self._hidden_channels = list(hidden_channels)

        first_out = hidden_channels[0] if hidden_channels else n_charges
        # Intermediate key only needed when a scalar MLP follows.
        self._intermediate_key = (
            f'{data_key_out}_intermediate' if hidden_channels else data_key_out
        )
        self.first_linear = IrrepsLinear(
            irreps_in=irreps_in,
            irreps_out=Irreps(f'{first_out}x0e'),
            data_key_in=data_key_in,
            data_key_out=self._intermediate_key,
            biases=False,
        )

        scalar_layers: list[nn.Module] = []
        if hidden_channels:
            dims = hidden_channels + [n_charges]
            for i in range(len(dims) - 1):
                scalar_layers.append(nn.SiLU())
                scalar_layers.append(nn.Linear(dims[i], dims[i + 1], bias=False))
        self.scalar_mlp = nn.Sequential(*scalar_layers)

        self._zero_init = zero_init
        if zero_init:
            for m in self.scalar_mlp.modules():
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.weight)

    @property
    def layer_instantiated(self) -> bool:
        # AtomGraphSequential._instantiate_modules only walks top-level modules,
        # so we expose the inner IrrepsLinear's lazy-instantiation status here.
        return self.first_linear.layer_instantiated

    def instantiate(self) -> None:
        self.first_linear.instantiate()
        if self._zero_init:
            nn.init.zeros_(self.first_linear.linear.weight)

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        data = self.first_linear(data)
        if self._hidden_channels:
            data[self.key_output] = self.scalar_mlp(data[self._intermediate_key])
        return data


class LatentEwaldSum(nn.Module):
    """
    Computes long-range energy via Ewald summation on latent charges.

    Sets up two differentiable leaves for LESForceStressOutput:

    POS (KEY.POS):
        requires_grad is enabled so d(E_LR)/d(pos) gives direct Ewald forces.

    LES_STRAIN (KEY.LES_STRAIN):
        Zero (n_graphs, 3, 3) leaf. Its symmetric part is applied to both pos
        and cell before les(), so d(E_LR)/d(les_strain) captures the complete
        Ewald stress (positional virial + cell contribution) in one autograd
        call. Formula: σ_lr = -(1/V) * strain_grad.

    Args:
        les_args:         kwargs forwarded to Les().
        data_key_in:      per-atom latent charges (N_atoms, 1).
        data_key_out:     per-graph LR energy output.
        compute_bec:      if True, compute Born effective charges.
        bec_output_index: 0/1/2 for x/y/z component of BEC.
    """

    def __init__(
        self,
        les_args: Optional[dict] = None,
        data_key_in: str = KEY.LES_Q,
        data_key_out: str = KEY.LR_ENERGY,
        compute_bec: bool = False,
        bec_output_index: Optional[int] = None,
    ):
        super().__init__()
        try:
            from les import Les  # https://github.com/ChengUCB/les
        except ImportError as e:
            raise ImportError(
                "The 'les' package is required for LES support. "
                'Install it with: pip install git+https://github.com/ChengUCB/les.git'
            ) from e

        if les_args is None:
            les_args = {'use_atomwise': False}
        self.key_input = data_key_in
        self.key_output = data_key_out
        self.compute_bec = compute_bec
        self.bec_output_index = bec_output_index
        self.les = Les(les_args)
        self._is_batch_data = True  # set by AtomGraphSequential.set_is_batch_data()

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        q = data[self.key_input]   # (N_atoms, n_charges)
        pos = data[KEY.POS]        # (N_atoms, 3)

        # Enable pos gradients for direct Ewald force (Path 2).
        if torch.is_grad_enabled() and pos.is_leaf and not pos.requires_grad:
            pos.requires_grad_(True)

        if self._is_batch_data:
            batch = data[KEY.BATCH].long()
            n_graphs = int(batch.max().item()) + 1
        else:
            batch = torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device)
            n_graphs = 1

        # Batched cell: SevenNet stores (3,3) per graph; PyG stacks to (3*n,3).
        if KEY.CELL in data:
            cell = data[KEY.CELL].view(-1, 3, 3)  # (n_graphs, 3, 3)
        else:
            cell = torch.zeros((n_graphs, 3, 3), device=pos.device, dtype=pos.dtype)

        if torch.is_grad_enabled():
            # Strain leaf for LR stress (Path 3).
            # Apply symmetric strain to pos and cell so that
            # d(E_LR)/d(les_strain) gives the complete affine-deformation
            # Ewald stress in one autograd call.
            les_strain = torch.zeros(
                (n_graphs, 3, 3), dtype=pos.dtype, device=pos.device,
            )
            les_strain.requires_grad_(True)
            data[KEY.LES_STRAIN] = les_strain

            sym_strain = 0.5 * (les_strain + les_strain.transpose(-1, -2))
            pos = pos + torch.bmm(pos.unsqueeze(-2), sym_strain[batch]).squeeze(-2)
            cell = cell + torch.bmm(cell, sym_strain)

        les_result = self.les(
            latent_charges=q,
            positions=pos,
            batch=batch,
            cell=cell,
            compute_energy=True,
            compute_bec=self.compute_bec,
            bec_output_index=self.bec_output_index,
        )

        e_lr = les_result['E_lr']  # (n_graphs,)
        assert e_lr is not None

        # Non-batch mode: squeeze to scalar to match SR_ENERGY from AtomReduce.
        data[self.key_output] = e_lr if self._is_batch_data else e_lr.squeeze()

        if self.compute_bec:
            bec = les_result.get('BEC')
            if bec is not None:
                data[KEY.LES_BEC] = bec

        return data


class NeutralizeCharge(nn.Module):
    """
    Enforce Σ_i q_i = 0 per graph (charge-neutrality constraint).

    Parameter-free post-processing on KEY.LES_Q. Applied right after
    LatentChargeReadout, before LatentEwaldSum. Gradient flows through
    arithmetic so backprop is normal.

    Args:
        mode:
            'none'  → identity (skip; module not normally inserted)
            'shift' → q_i ← q_i − ⟨q⟩_graph             (uniform shift)
            'fukui' → q_i ← q_i − (Σq) · softplus(f_i) / Σ_j softplus(f_j)
                      (Fukui-style redistribution; needs KEY.LES_F input)
        data_key_q: per-atom latent charges (N, n_charges)
        data_key_f: per-atom Fukui factor (N, 1)  [only for 'fukui']
        eps:        denominator guard
    """

    def __init__(
        self,
        mode: str = 'none',
        data_key_q: str = KEY.LES_Q,
        data_key_f: str = KEY.LES_F,
        eps: float = 1e-12,
    ):
        super().__init__()
        if mode not in ('none', 'shift', 'fukui'):
            raise ValueError(
                f"Unknown neutralize_mode: {mode!r}. "
                "Choose from 'none' | 'shift' | 'fukui'."
            )
        self.mode = mode
        self.data_key_q = data_key_q
        self.data_key_f = data_key_f
        self.eps = eps
        self._is_batch_data = True  # set by AtomGraphSequential.set_is_batch_data

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        if self.mode == 'none':
            return data

        q = data[self.data_key_q]   # (N_atoms, n_charges)
        if self._is_batch_data:
            batch = data[KEY.BATCH].long()
            n_graphs = int(batch.max().item()) + 1
        else:
            batch = torch.zeros(q.shape[0], dtype=torch.long, device=q.device)
            n_graphs = 1

        n_ch = q.shape[1]

        if self.mode == 'shift':
            # per-graph mean
            sum_q = torch.zeros(n_graphs, n_ch, device=q.device, dtype=q.dtype)
            sum_q.scatter_add_(
                0, batch.unsqueeze(-1).expand(-1, n_ch), q
            )
            ones = torch.ones(q.shape[0], device=q.device, dtype=q.dtype)
            count = torch.zeros(n_graphs, device=q.device, dtype=q.dtype)
            count.scatter_add_(0, batch, ones)
            mean_q = sum_q / count.clamp(min=1.0).unsqueeze(-1)
            q = q - mean_q[batch]

        elif self.mode == 'fukui':
            f_raw = data[self.data_key_f]   # expected (N, 1) or (N,)
            if f_raw.dim() == 1:
                f_raw = f_raw.unsqueeze(-1)
            f = torch.nn.functional.softplus(f_raw)   # (N, 1) positive

            f_sum = torch.zeros(n_graphs, 1, device=q.device, dtype=q.dtype)
            f_sum.scatter_add_(0, batch.unsqueeze(-1), f)
            q_sum = torch.zeros(n_graphs, n_ch, device=q.device, dtype=q.dtype)
            q_sum.scatter_add_(
                0, batch.unsqueeze(-1).expand(-1, n_ch), q
            )
            # ratio (N, 1) broadcasts over n_ch
            ratio = f / (f_sum[batch] + self.eps)
            q = q - q_sum[batch] * ratio

        data[self.data_key_q] = q
        return data


class AddLREnergy(nn.Module):
    """Adds LR energy to SR energy: PRED_TOTAL_ENERGY = SR_ENERGY + LR_ENERGY."""

    def __init__(
        self,
        key_sr: str = KEY.SR_ENERGY,
        key_lr: str = KEY.LR_ENERGY,
        data_key_out: str = KEY.PRED_TOTAL_ENERGY,
    ):
        super().__init__()
        self.key_sr = key_sr
        self.key_lr = key_lr
        self.key_output = data_key_out

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        data[self.key_output] = data[self.key_sr] + data[self.key_lr]
        return data


class DipoleCorrection(nn.Module):
    """
    Bengtsson-style slab dipole correction driven by LES latent charges.

    Per graph:
        μ_axis = Σ_i q_i · r_{i,axis}            (e·Å, summed over channels)
        E_dip  = sign · μ² / (2 ε₀ V_cell)        (eV)

    With ``sign = -1`` (default) and raw DFT data containing the PBC dipole-
    image artifact, this matches the artifact baked into the data, so that the
    loss compares like-with-like. This is the in-model counterpart of
    ``correct_dataset.py``'s dataset-side ``+μ²/(2ε₀V)`` correction:

        (A) dataset E += +μ²/(2ε₀V)              model has no DipoleCorrection
        (B) dataset E unchanged                  model adds  -μ²/(2ε₀V) here

    Use (A) XOR (B) — applying both is double counting.

    Forces are produced automatically by ``LESForceStressOutput``:
      - direct ∂E_dip/∂pos contributes to F via the position grad path
      - charge-readout ∂E_dip/∂edge_vec contributes via the edge grad path

    Stress contribution from the explicit z-dependence is **not** captured
    (we don't tap into ``LES_STRAIN``); only the q-path edge virial flows
    through. For slab systems this is usually acceptable since stress is
    typically not used in the loss for dipole-laden geometries.

    Args:
        axis:  cartesian axis (0|1|2) along which the slab dipole lives.
               default 2 (z).
        sign:  ±1 multiplier. -1 reproduces raw DFT (default, matches (B)).
        eps0:  vacuum permittivity in e²/(eV·Å); default 5.5263499562e-3.
        data_key_q: latent charges input (N, n_charges).
        data_key_e: PRED_TOTAL_ENERGY (in-place add).
    """

    EPS0_AU: float = 5.5263499562e-3

    def __init__(
        self,
        axis: int = 2,
        sign: float = -1.0,
        eps0: Optional[float] = None,
        data_key_q: str = KEY.LES_Q,
        data_key_e: str = KEY.PRED_TOTAL_ENERGY,
    ):
        super().__init__()
        if axis not in (0, 1, 2):
            raise ValueError(f'axis must be 0|1|2; got {axis}')
        if sign not in (-1.0, 1.0):
            # not strictly enforced, but warn-ish via assert
            pass
        self.axis = int(axis)
        self.sign = float(sign)
        self.eps0 = float(eps0) if eps0 is not None else self.EPS0_AU
        self.key_q = data_key_q
        self.key_e = data_key_e
        self._is_batch_data = True  # set by AtomGraphSequential.set_is_batch_data

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        q = data[self.key_q]          # (N, n_charges)
        pos = data[KEY.POS]           # (N, 3)

        # Make sure positions track gradients (LESForceStressOutput will read
        # d(E)/d(pos) — needed even if les_lr_energy already enabled it).
        if torch.is_grad_enabled() and pos.is_leaf and not pos.requires_grad:
            pos.requires_grad_(True)

        if self._is_batch_data:
            batch = data[KEY.BATCH].long()
            n_graphs = int(batch.max().item()) + 1
        else:
            batch = torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device)
            n_graphs = 1

        n_ch = q.shape[1]
        r_axis = pos[:, self.axis]                       # (N,)
        qz = q * r_axis.unsqueeze(-1)                    # (N, n_ch)
        mu = torch.zeros(n_graphs, n_ch, device=q.device, dtype=q.dtype)
        mu.scatter_add_(
            0, batch.unsqueeze(-1).expand(-1, n_ch), qz
        )                                                # (n_graphs, n_ch)

        # Cell volume per graph
        if KEY.CELL in data:
            cell = data[KEY.CELL].view(-1, 3, 3)
            V = torch.det(cell).abs()                    # (n_graphs,)
        else:
            V = data[KEY.CELL_VOLUME]

        e_dip_per_graph = (
            self.sign
            * (mu * mu).sum(dim=-1)
            / (2.0 * self.eps0 * V.clamp(min=1e-12))
        )                                                # (n_graphs,)

        e_tot = data[self.key_e]
        if self._is_batch_data:
            data[self.key_e] = e_tot + e_dip_per_graph
        else:
            data[self.key_e] = e_tot + e_dip_per_graph.squeeze()
        return data


class LESForceStressOutput(nn.Module):
    """
    Force and stress output for LES models. Replaces ForceStressOutputFromEdge.

    Three gradient paths in a single torch.autograd.grad call:
      Path 1  d(E_total)/d(EDGE_VEC)  SR + q-path LR forces; edge virial stress
      Path 2  d(E_LR)/d(POS)          direct Ewald positional forces
      Path 3  d(E_LR)/d(LES_STRAIN)   complete Ewald stress (pos + cell)

    All three are computed in one call because separate calls would free the
    graph after Path 1 (retain_graph=False when create_graph=False at inference),
    causing Path 2/3 to fail.

    Atomic virial is not supported: direct Ewald forces (Path 2) depend on
    absolute positions and have no pairwise decomposition.
    """

    def __init__(
        self,
        data_key_edge: str = KEY.EDGE_VEC,
        data_key_edge_idx: str = KEY.EDGE_IDX,
        data_key_energy: str = KEY.PRED_TOTAL_ENERGY,
        data_key_pos: str = KEY.POS,
        data_key_force: str = KEY.PRED_FORCE,
        data_key_stress: str = KEY.PRED_STRESS,
        data_key_cell_volume: str = KEY.CELL_VOLUME,
        use_atomic_virial: bool = False,
    ):
        super().__init__()
        if use_atomic_virial:
            raise NotImplementedError(
                'Atomic virial is not supported for LES models. '
                'Direct Ewald forces (Path 2) have no pairwise decomposition. '
                'Use total stress instead.'
            )
        self.key_edge = data_key_edge
        self.key_edge_idx = data_key_edge_idx
        self.key_energy = data_key_energy
        self.key_pos = data_key_pos
        self.key_force = data_key_force
        self.key_stress = data_key_stress
        self.key_cell_volume = data_key_cell_volume
        self.use_atomic_virial = False
        self._is_batch_data = True  # set by AtomGraphSequential.set_is_batch_data()

    def get_grad_key(self) -> str:
        return self.key_edge

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        tot_num = torch.sum(data[KEY.NUM_ATOMS])
        rij = data[self.key_edge]
        energy = [(data[self.key_energy]).sum()]
        edge_idx = data[self.key_edge_idx]
        pos = data[self.key_pos]

        has_les_strain = KEY.LES_STRAIN in data
        grad_inputs = [rij, pos]
        if has_les_strain:
            grad_inputs.append(data[KEY.LES_STRAIN])

        grads = torch.autograd.grad(
            energy,
            grad_inputs,
            create_graph=self.training,
            allow_unused=True,
        )

        fij = grads[0]          # d(E_total)/d(EDGE_VEC): SR + q-path LR forces
        pos_grad = grads[1]     # d(E_LR)/d(POS): direct Ewald forces
        strain_grad = grads[2] if has_les_strain else None  # d(E_LR)/d(LES_STRAIN): Ewald stress

        force = torch.zeros(tot_num, 3, dtype=rij.dtype, device=rij.device)

        # ── Path 1: edge gradient → forces + edge virial stress ──
        if fij is not None:
            pf = torch.zeros(tot_num, 3, dtype=fij.dtype, device=fij.device)
            nf = torch.zeros(tot_num, 3, dtype=fij.dtype, device=fij.device)
            _edge_src = broadcast(edge_idx[0], fij, 0)
            _edge_dst = broadcast(edge_idx[1], fij, 0)
            pf.scatter_reduce_(0, _edge_src, fij, reduce='sum')
            nf.scatter_reduce_(0, _edge_dst, fij, reduce='sum')
            force = pf - nf

            diag = rij * fij
            s12 = rij[..., 0] * fij[..., 1]
            s23 = rij[..., 1] * fij[..., 2]
            s31 = rij[..., 2] * fij[..., 0]
            _virial = torch.cat(
                [diag, s12.unsqueeze(-1), s23.unsqueeze(-1), s31.unsqueeze(-1)],
                dim=-1,
            )
            _s = torch.zeros(tot_num, 6, dtype=fij.dtype, device=fij.device)
            _edge_dst6 = broadcast(edge_idx[1], _virial, 0)
            _s.scatter_reduce_(0, _edge_dst6, _virial, reduce='sum')

            if self._is_batch_data:
                batch = data[KEY.BATCH]
                nbatch = int(batch.max().cpu().item()) + 1
                sout = torch.zeros(
                    (nbatch, 6), dtype=_virial.dtype, device=_virial.device
                )
                _batch = broadcast(batch, _s, 0)
                sout.scatter_reduce_(0, _batch, _s, reduce='sum')
            else:
                sout = torch.sum(_s, dim=0)

            data[self.key_stress] = (
                torch.neg(sout) / data[self.key_cell_volume].unsqueeze(-1)
            )

        # ── Path 2: position gradient → direct Ewald forces ──
        if pos_grad is not None:
            force = force - pos_grad

        data[self.key_force] = force

        # ── Path 3: strain gradient → complete LR stress ──
        # σ_lr = -(1/V) * strain_grad  (affine deformation: pos + cell both strained)
        if strain_grad is not None:
            volume = data[self.key_cell_volume]
            if self._is_batch_data:
                lr_stress_3x3 = (
                    torch.neg(strain_grad) / volume.unsqueeze(-1).unsqueeze(-1)
                )
                lr_stress_voigt = torch.stack(
                    [
                        lr_stress_3x3[:, 0, 0],
                        lr_stress_3x3[:, 1, 1],
                        lr_stress_3x3[:, 2, 2],
                        lr_stress_3x3[:, 0, 1],
                        lr_stress_3x3[:, 1, 2],
                        lr_stress_3x3[:, 0, 2],
                    ],
                    dim=-1,
                )  # (n_graphs, 6)
            else:
                lr_stress_3x3 = torch.neg(strain_grad.squeeze(0)) / volume  # (3, 3)
                lr_stress_voigt = torch.stack(
                    [
                        lr_stress_3x3[0, 0],
                        lr_stress_3x3[1, 1],
                        lr_stress_3x3[2, 2],
                        lr_stress_3x3[0, 1],
                        lr_stress_3x3[1, 2],
                        lr_stress_3x3[0, 2],
                    ]
                )  # (6,)

            if self.key_stress in data:
                data[self.key_stress] = data[self.key_stress] + lr_stress_voigt
            else:
                data[self.key_stress] = lr_stress_voigt

        return data

from typing import Literal, Optional, Tuple, List, Dict, Iterable
import itertools
from abc import ABC, abstractmethod


import torch
from torch.autograd.functional import jacobian

from ..model.model_param_estimator import BaseBuckEstimator
from ..parameters.parameter_class import Parameters, Value


class JacobianEstimatorBase(ABC):

    @staticmethod
    def _seq_keys_and_lengths(params: "Parameters") -> Tuple[List[str], List[int]]:
        """
        Return (seq_keys, lengths) for parameters that are sequences.
        Order matches Parameters.expand_torch_sequences() (i.e., appearance order).
        """
        seq_keys: List[str] = []
        lengths:  List[int] = []
        for k, v in params.params.items():
            if isinstance(v, (list, tuple)):
                seq_keys.append(k)
                lengths.append(len(v))
            elif isinstance(v, torch.Tensor) and v.ndim >= 1:
                seq_keys.append(k)
                lengths.append(int(v.shape[0]))
        return seq_keys, lengths

    @staticmethod
    def _fix_params_at_index(params: Parameters, grid_idx: Tuple[int, ...]) -> Parameters:
        """
        Build a new Parameters where every sequence-like parameter is replaced
        by the *single* element at the corresponding index in grid_idx.
        Non-sequence params are copied unchanged.

        If there are k sequence params, grid_idx has length k and its order
        matches _seq_keys_and_lengths().
        """
        seq_keys, _ = JacobianEstimator._seq_keys_and_lengths(params)
        if len(seq_keys) != len(grid_idx):
            raise ValueError(
                f"Grid index length {len(grid_idx)} does not match number of sequence params {len(seq_keys)}."
            )

        new_dict: Dict[str, Value] = {}
        # First, copy everything
        for k, v in params.params.items():
            new_dict[k] = v

        # Then, replace sequence params with their selected element
        for ax, k in enumerate(seq_keys):
            v = params.params[k]
            idx = grid_idx[ax]
            if isinstance(v, (list, tuple)):
                elem = v[idx]
                new_dict[k] = elem  # scalar/tensor element
            elif isinstance(v, torch.Tensor) and v.ndim >= 1:
                elem = v[idx]  # slice along leading dim
                new_dict[k] = elem
            else:
                # Should not happen: seq_keys ensures v is sequence-like
                new_dict[k] = v

        return type(params)(**new_dict)

    @classmethod
    @abstractmethod
    def estimate_Jacobian(cls, X: torch.Tensor, model: BaseBuckEstimator, *args, **kwargs) -> torch.Tensor:
        """Estimate the Jacobian of the model forward pass with respect to its inputs."""
        ...


class JacobianEstimator(JacobianEstimatorBase):
    ## -------- single-time-index Jacobian on a (B,F) slice --------
    @staticmethod
    def estimate_J_single_parameter_combination(
        Xi: torch.Tensor,  # (B, F) for one specific grid slice
        t: int,  # 0 .. B-2
        model: BaseBuckEstimator,  # estimator; must accept (B,1,F)
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Compute local Jacobian of one-step map at time t for a single (B,F) series.

        Returns:
            J : (S, F)  where S is the number of output state quantities
        """
        vec = Xi[t].clone().to(dtype).requires_grad_(True)

        def local_fwd(v):
            Xi_mod = Xi.clone()
            Xi_mod[t] = v
            fwd = model(Xi_mod.unsqueeze(1))  # expect (B-1, 1, S)
            # pick the output at the same time index t
            fwd_t = fwd[t].squeeze(0)  # (S,)
            return fwd_t

        J = torch.autograd.functional.jacobian(local_fwd, vec, create_graph=False)  # (S, F)
        return J

    # -------- full Jacobian across the parameter grid --------
    @classmethod
    def estimate_Jacobian(
        cls,
        X: torch.Tensor,  # shape (B, t1, t2, ..., F)
        model: BaseBuckEstimator,  # estimator with get_estimates()
        *,
        by_series: bool = True,  # kept for compatibility; irrelevant with multi-d grid
        number_of_samples: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
        rng: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Generalized Jacobian of the one-step RK4 map w.r.t. input X.

        The parameter-dependent transient grid comes from the *middle axes* of X:
            X.shape = (B, t1, t2, ..., F)

        For each grid index (i1, i2, ...), we:
            1) slice Xi = X[:, i1, i2, ..., :]  -> (B, F)
            2) build a fixed-parameter clone at that grid point
            3) average local Jacobians over sampled time indices

        Returns:
            out : tensor of shape (t1, t2, ..., F, S)
        """
        device = X.device
        B = X.shape[0]
        F = X.shape[-1]
        grid_shape = tuple(X.shape[1:-1])  # (t1, t2, ...)
        grid_nd = len(grid_shape)

        if B < 2:
            raise ValueError("Need at least 2 time rows to form one-step Jacobian (B>=2).")

        # --- choose time indices to sample
        max_idx = B - 2
        n_samples = (
            max_idx + 1 if number_of_samples is None else min(number_of_samples, max_idx + 1)
        )
        if n_samples == max_idx + 1:
            t_samples = torch.arange(n_samples, device=device)
        else:
            g = rng if rng is not None else torch.default_generator
            t_samples = torch.randperm(max_idx + 1, generator=g, device=device)[:n_samples]

        # --- determine output dimension S by a quick forward on one slice
        # pick first grid slice (or the only one)
        first_grid_idx = (0,) * grid_nd if grid_nd > 0 else ()
        Xi0 = X[(slice(None), *first_grid_idx, slice(None))]  # (B, F)
        # quick run
        with torch.no_grad():
            y0 = model(Xi0.unsqueeze(1))  # (B-1, 1, S)
        S = int(y0.shape[-1])

        # --- allocate output: (t1, t2, ..., F, S)
        out = torch.zeros(*grid_shape, F, S, dtype=dtype, device=device)

        # --- base params and their sequence layout
        base_params = model.get_estimates()  # Parameters (physical units)

        # --- loop over parameter grid
        grid_ranges: Iterable[range] = [range(n) for n in grid_shape] if grid_nd > 0 else [range(1)]
        for grid_idx in itertools.product(*grid_ranges):
            if grid_nd == 0:
                grid_idx = ()

            # 1) pick input slice
            Xi = X[(slice(None), *grid_idx, slice(None))].to(dtype).detach()  # (B, F)

            # 2) fix parameters at this grid point and clone the model
            fixed_params = cls._fix_params_at_index(base_params, grid_idx)
            model_clone = type(model)(param_init=fixed_params).to(device=device, dtype=dtype)
            model_clone.eval()

            # 3) accumulate Jacobians across sampled times
            Jsum = torch.zeros(S, F, dtype=dtype, device=device)
            for t in t_samples.tolist():
                Ji = cls.estimate_J_single_parameter_combination(
                    Xi, t, model_clone, dtype=dtype
                )  # (S, F)
                Jsum += Ji

            Jmean = (Jsum / float(n_samples)).transpose(0, 1)  # -> (F, S)
            out[grid_idx] = Jmean

        # rotate the grid so that the last two axes are (S, F): J = ∂y/∂x
        out = out.mT

        # Compatibility path (by_series flag kept for API parity):
        # with multi-d grids, by_series=False would mean averaging across all grid points.
        if not by_series and grid_nd > 0:
            # average over all grid axes -> (F, S)
            return out.mean(dim=tuple(range(0, grid_nd)))

        return out


class FwdBckJacobianEstimator(JacobianEstimatorBase):
    @staticmethod
    def _build_two_row_vec(x_row_next, x_row_this):
        # order: [i_{t+1}, v_{t+1}, D_t, Δt_t]
        return torch.stack([x_row_next[0], x_row_next[1], x_row_this[2], x_row_this[3]])

    @classmethod
    def estimate_J_single_parameter_combination(
        cls, Xi, t, model: BaseBuckEstimator, direction: Literal["forward", "backward"], dtype=torch.float32
    ):
        """
        Xi : (B,4) for one series
        t  : time index (0 … B-2)
        returns (2,4) Jacobian for that t
        """        

        def local_fwd(v):
            Xi_mod = Xi.clone()
            Xi_mod[t] = v  # plug variable slice
            fwd, _ = model(Xi_mod.unsqueeze(1))  # (B-1,1,2)
            return fwd[t].squeeze(0)  # (2,)

        if direction == "forward":
            vec = Xi[t].clone().to(dtype).requires_grad_(True)
            return torch.autograd.functional.jacobian(local_fwd, vec, create_graph=False)

        elif direction == "backward":
            x_next = Xi[t + 1].detach()
            x_this = Xi[t].detach()

            vec = cls._build_two_row_vec(x_next, x_this).to(dtype).requires_grad_(True)  # shape (4,)

            def _unpack(vec):
                # inverse of _build_two_row_vec
                x_next_mod = torch.stack([vec[0], vec[1], x_next[2], x_next[3]])
                x_this_mod = torch.stack([x_this[0], x_this[1], vec[2], vec[3]])
                Xi_mod = Xi.clone()
                Xi_mod[t + 1] = x_next_mod
                Xi_mod[t] = x_this_mod
                return Xi_mod

            def local_bck(v):
                Xi_mod = _unpack(v)
                _, bck = model(Xi_mod.unsqueeze(1))  # (B-1,1,2)
                return bck[t].squeeze(0)  # (2,)

            return torch.autograd.functional.jacobian(local_bck, vec, create_graph=False)  # (2,4)
        else:
            raise ValueError(f"Unknown direction {direction}, expected 'forward' or 'backward'.")

    @classmethod
    def estimate_Jacobian(
        cls,
        X: torch.Tensor,  # shape (B, t1, t2, ..., F)
        model: BaseBuckEstimator,  # estimator with get_estimates()
        *,
        by_series: bool = True,  # kept for compatibility; irrelevant with multi-d grid
        number_of_samples: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
        rng: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Generalized Jacobian of the one-step RK4 map w.r.t. input X.

        The parameter-dependent transient grid comes from the *middle axes* of X:
            X.shape = (B, t1, t2, ..., F)

        For each grid index (i1, i2, ...), we:
            1) slice Xi = X[:, i1, i2, ..., :]  -> (B, F)
            2) build a fixed-parameter clone at that grid point
            3) average local Jacobians over sampled time indices

        Returns:
            out : tensor of shape (t1, t2, ..., F, S)
        """
        device = X.device
        B = X.shape[0]
        F = X.shape[-1]
        grid_shape = tuple(X.shape[1:-1])  # (t1, t2, ...)
        grid_nd = len(grid_shape)

        if B < 2:
            raise ValueError("Need at least 2 time rows to form one-step Jacobian (B>=2).")

        # --- choose time indices to sample
        max_idx = B - 2
        n_samples = (
            max_idx + 1 if number_of_samples is None else min(number_of_samples, max_idx + 1)
        )
        if n_samples == max_idx + 1:
            t_samples = torch.arange(n_samples, device=device)
        else:
            g = rng if rng is not None else torch.default_generator
            t_samples = torch.randperm(max_idx + 1, generator=g, device=device)[:n_samples]

        # --- determine output dimension S by a quick forward on one slice
        # pick first grid slice (or the only one)
        first_grid_idx = (0,) * grid_nd if grid_nd > 0 else ()
        Xi0 = X[(slice(None), *first_grid_idx, slice(None))]  # (B, F)
        # quick run
        with torch.no_grad():
            y0 = model(Xi0.unsqueeze(1))  # (2, B-1, 1, S)
        S = int(y0.shape[-1])

        # --- allocate output: (2, t1, t2, ..., F, S)
        out = torch.zeros(2, *grid_shape, F, S, dtype=dtype, device=device)

        # --- base params and their sequence layout
        base_params = model.get_estimates()  # Parameters (physical units)

        # --- loop over parameter grid
        grid_ranges: Iterable[range] = [range(n) for n in grid_shape] if grid_nd > 0 else [range(1)]
        for grid_idx in itertools.product(*grid_ranges):
            if grid_nd == 0:
                grid_idx = ()

            # 1) pick input slice
            Xi = X[(slice(None), *grid_idx, slice(None))].to(dtype).detach()  # (B, F)

            # 2) fix parameters at this grid point and clone the model
            fixed_params = cls._fix_params_at_index(base_params, grid_idx)
            model_clone = type(model)(param_init=fixed_params).to(device=device, dtype=dtype)
            model_clone.eval()

            # 3) accumulate Jacobians across sampled times
            Jsum = torch.zeros(2, S, F, dtype=dtype, device=device)
            for t in t_samples.tolist():
                Ji_fwd = cls.estimate_J_single_parameter_combination(
                    Xi, t, model_clone, direction="forward", dtype=dtype
                )  # (S, F)
                Ji_bck = cls.estimate_J_single_parameter_combination(
                    Xi, t, model_clone, direction="backward", dtype=dtype
                )  # (S, F)

                Ji = torch.stack([Ji_fwd, Ji_bck], dim=0)  # shape (2, S, F)
                Jsum += Ji

            Jmean = (Jsum / float(n_samples)).transpose(1, 2)  # -> (2, F, S)
            out[(slice(None),) + grid_idx + (Ellipsis,)] = Jmean

        # rotate the grid so that the last two axes are (S, F): J = ∂y/∂x
        out = out.mT

        # Compatibility path (by_series flag kept for API parity):
        # with multi-d grids, by_series=False would mean averaging across all grid points.
        if not by_series and grid_nd > 0:
            # average over all grid axes -> (F, S)
            return out.mean(dim=tuple(range(0, grid_nd)))

        return out

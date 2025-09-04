from typing import Literal, Optional
from abc import ABC, abstractmethod


import torch
from torch.autograd.functional import jacobian

from ..model.model_param_estimator import BaseBuckEstimator


class JacobianEstimatorBase(ABC):
    @classmethod
    @abstractmethod
    def estimate_Jacobian(cls, X: torch.Tensor, model: BaseBuckEstimator, *args, **kwargs) -> torch.Tensor:
        """Estimate the Jacobian of the model forward pass with respect to its inputs."""
        ...


class JacobianEstimator(JacobianEstimatorBase):
    @classmethod
    def estimate_J_single_t(
        cls, Xi: torch.Tensor, t: int, model: BaseBuckEstimator, dtype=torch.float32
    ) -> torch.Tensor:
        """
        Xi : (B,4) for one series
        t  : time index (0 … B-2)
        returns (2,4) Jacobian for that t
        """
        vec = Xi[t].clone().to(dtype).requires_grad_(True)

        def local_fwd(v):
            Xi_mod = Xi.clone()
            Xi_mod[t] = v  # plug variable slice
            fwd = model(Xi_mod.unsqueeze(1))  # (B-1,1,2)
            return fwd[t].squeeze(0)  # (2,)

        J = torch.autograd.functional.jacobian(local_fwd, vec, create_graph=False)  # (2,4)
        return J

    @classmethod
    def estimate_Jacobian(
        cls,
        X: torch.Tensor,  # (B, T, 4)
        model: BaseBuckEstimator,
        *,
        by_series: bool = True,
        number_of_samples: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
        rng: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Estimate the Jacobian of the RK4 state map.

        Parameters
        ----------
        model
            A *trained* ``BaseBuckEstimator`` (or any compatible subclass).
        X
            Input tensor of shape ``(B, T, 4)`` where *B* is number of time
            rows and *T* is number of independent series.
        by_series
            If *True*, return an array of shape ``(T, 2, 4)`` with one
            Jacobian per series.  Otherwise return their mean ``(2,4)``.
        number_of_samples
            Maximum number of time indices *t* to sample per series.  If
            *None* uses *all* valid indices ``0 … B-2``.
        dtype
            Promote *all* inputs to this dtype before computing Jacobians.
        rng
            Optional ``torch.Generator`` for deterministic sub‑sampling.

        Returns
        -------
        Tensor
            ``(2,4)`` if ``by_series=False`` else ``(T,2,4)``.
        """

        B, T, _ = X.shape
        max_idx = B - 2  # last valid t
        number_of_samples = (
            max_idx + 1 if number_of_samples is None else min(number_of_samples, max_idx + 1)
        )

        if number_of_samples == max_idx + 1:
            t_samples = torch.arange(number_of_samples, device=X.device)
        else:
            g = rng if rng is not None else torch.default_generator
            t_samples = torch.randperm(max_idx + 1, generator=g, device=X.device)[
                :number_of_samples
            ]

        J_series: list[torch.Tensor] = []

        for s in range(T):
            Xi = X[:, s].to(dtype).detach()  # (B,4)  independent series s
            Jsum = torch.zeros(2, 4, dtype=dtype, device=X.device)
            # update the model so that the load resistance is one and fixed to the value of the t-th load
            model_params_T = model.get_estimates()
            # can't directly set Parameter attributes because they are NamedTuples
            model_params_T.Rloads = [model_params_T.Rloads[s]]  # fix to the s-th load
            model_class = model.__class__
            model_clone = model_class(param_init=model_params_T)

            model_clone = model_clone.to(dtype=dtype, device=X.device)
            model_clone.eval()

            for t in t_samples:
                # choose the time-slice we differentiate with respect to
                Ji = cls.estimate_J_single_t(Xi, t, model_clone, dtype=dtype)
                Jsum += Ji
            J_series.append(Jsum / number_of_samples)

        J_stack = torch.stack(J_series)  # (T,2,4)
        return J_stack if by_series else J_stack.mean(dim=0)


class FwdBckJacobianEstimator(JacobianEstimatorBase):
    @staticmethod
    def _build_two_row_vec(x_row_next, x_row_this):
        # order: [i_{t+1}, v_{t+1}, D_t, Δt_t]
        return torch.stack([x_row_next[0], x_row_next[1], x_row_this[2], x_row_this[3]])

    @classmethod
    def estimate_bck_J_single_t(cls, Xi, t, model: BaseBuckEstimator, dtype=torch.float32):
        """
        Xi : (B,4) for one series
        t  : time index (0 … B-2)
        returns (2,4) Jacobian for that t
        """
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

        J = torch.autograd.functional.jacobian(local_bck, vec, create_graph=False)  # (2,4)
        return J

    @classmethod
    def estimate_fwd_J_single_t(
        cls, Xi: torch.Tensor, t: int, model: BaseBuckEstimator, dtype=torch.float32
    ) -> torch.Tensor:
        """
        Xi : (B,4) for one series
        t  : time index (0 … B-2)
        returns (2,4) Jacobian for that t
        """
        vec = Xi[t].clone().to(dtype).requires_grad_(True)

        def local_fwd(v):
            Xi_mod = Xi.clone()
            Xi_mod[t] = v  # plug variable slice
            fwd, _ = model(Xi_mod.unsqueeze(1))  # (B-1,1,2)
            return fwd[t].squeeze(0)  # (2,)

        J = torch.autograd.functional.jacobian(local_fwd, vec, create_graph=False)  # (2,4)
        return J

    def _check_model_compatibility(model: BaseBuckEstimator, X: torch.Tensor):
        if not isinstance(model, BaseBuckEstimator):
            raise TypeError(f"Expected model of type BaseBuckEstimator, got {type(model)}")
            # check if the number of dimensions is correct
        pred: torch.Tensor = model(X)
        if pred.dim() != 4:
            raise ValueError(f"Expected 4D output, got shape {pred.shape}")
        if pred.shape[0] != 2:
            raise ValueError(f"Expected 2 batches, got shape {pred.shape}")

    @classmethod
    def estimate_Jacobian(
        cls,
        X: torch.Tensor,  # (B, T, 4)
        model: BaseBuckEstimator,
        *,
        direction: Literal["forward", "backward"] = "forward",
        by_series: bool = True,
        number_of_samples: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
        rng: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Estimate the Jacobian of the RK4 state map.

        Parameters
        ----------
        model
            A *trained* ``BaseBuckEstimator`` (or any compatible subclass).
        X
            Input tensor of shape ``(B, T, 4)`` where *B* is number of time
            rows and *T* is number of independent series.
        direction
            ``"forward"`` → ∂state_{t+1}/∂inputs_t ;
            ``"backward"`` → ∂state_t/∂[state_{t+1}, controls_t].
        by_series
            If *True*, return an array of shape ``(T, 2, 4)`` with one
            Jacobian per series.  Otherwise return their mean ``(2,4)``.
        number_of_samples
            Maximum number of time indices *t* to sample per series.  If
            *None* uses *all* valid indices ``0 … B-2``.
        dtype
            Promote *all* inputs to this dtype before computing Jacobians.
        rng
            Optional ``torch.Generator`` for deterministic sub‑sampling.

        Returns
        -------
        Tensor
            ``(2,4)`` if ``by_series=False`` else ``(T,2,4)``.
        """

        cls._check_model_compatibility(model, X)

        B, T, _ = X.shape
        max_idx = B - 2  # last valid t
        number_of_samples = (
            max_idx + 1 if number_of_samples is None else min(number_of_samples, max_idx + 1)
        )

        if number_of_samples == max_idx + 1:
            t_samples = torch.arange(number_of_samples, device=X.device)
        else:
            g = rng if rng is not None else torch.default_generator
            t_samples = torch.randperm(max_idx + 1, generator=g, device=X.device)[:number_of_samples]

        J_series: list[torch.Tensor] = []

        for s in range(T):
            Xi = X[:, s].to(dtype).detach()  # (B,4)  independent series s
            Jsum = torch.zeros(2, 4, dtype=dtype, device=X.device)
            # update the model so that the load resistance is one and fixed to the value of the t-th load
            model_params_T = model.get_estimates()
            model_params_T.Rloads = [model_params_T.Rloads[s]]  # fix to the s-th load
            # can't directly set Parameter attributes because they are NamedTuples
            model_class = model.__class__
            model_clone = model_class(param_init=model_params_T)

            model_clone = model_clone.to(dtype=dtype, device=X.device)
            model_clone.eval()

            for t in t_samples:
                # choose the time-slice we differentiate with respect to
                if direction == "forward":
                    Ji = cls.estimate_fwd_J_single_t(Xi, t, model_clone, dtype=dtype)
                elif direction == "backward":
                    Ji = cls.estimate_bck_J_single_t(Xi, t, model_clone, dtype=dtype)
                Jsum += Ji
            J_series.append(Jsum / number_of_samples)

        J_stack = torch.stack(J_series)  # (T,2,4)
        return J_stack if by_series else J_stack.mean(dim=0)

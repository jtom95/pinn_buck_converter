import torch
from ..config import Parameters


def rel_tolerance_to_sigma(
    rel_tol: Parameters,
    number_of_stds_in_relative_tolerance: float = 2.96,
    dtype: torch.dtype = torch.float32,
) -> Parameters:
    """
    Map a ± δ *relative* tolerance in linear space to the *log-space*
    standard-deviation **σ_log** of a Log-Normal prior.

    Theory
    ------
    Let a positive parameter *z* follow a Log-Normal prior

    ``log z  ~  N( μ_log , σ_log² )``  ⇒  ``z  =  exp( μ_log  ±  k·σ_log )``

    If the manufacturer’s tolerance says
    “**the true value lies within ± δ of nominal with k σ confidence**”
    then the multiplicative factor at k σ must equal ``1 + δ``:

    ``exp(k · σ_log) = 1 + δ      ⇒      σ_log = ln(1 + δ) / k``

    * `δ`  – relative tolerance (e.g. 0.3 → ± 30 %)
    * `k`  – how many standard deviations that tolerance represents
             (`number_of_stds_in_relative_tolerance`, default **2.96**
             for a two-sided 95 % band).

    Parameters
    ----------
    rel_tol : Parameters
        Relative tolerances δ for every component (same structure as
        :class:`Parameters`).
    number_of_stds_in_relative_tolerance : float, default **2.96**
        The *k* in the formula above.
        Typical choices:

        * **1.0** → δ is a 68 % (1 σ) band
        * **1.96** → δ is a 95 % (Gaussian 2-sided) band
        * **2.58** → δ is a 99 % band
        * **2.96** → δ is a 2-sided 95 % band for a Log-Normal
    dtype : torch.dtype
        Desired dtype of the returned σ tensors.

    Returns
    -------
    Parameters
        A new :class:`Parameters` instance holding **σ_log** for every field.

    Notes
    -----
    * ``torch.log1p(δ)`` is numerically stable for small δ.
    * The function leaves the *shape* of the `Parameters` object intact,
      only replacing each value with its corresponding σ.
    """

    def _to_sigma(delta: float) -> torch.Tensor:
        # σ_log = ln(1 + δ) / k
        return torch.log1p(torch.tensor(delta, dtype=dtype)) / number_of_stds_in_relative_tolerance

    return Parameters(
        L=_to_sigma(rel_tol.L),
        RL=_to_sigma(rel_tol.RL),
        C=_to_sigma(rel_tol.C),
        RC=_to_sigma(rel_tol.RC),
        Rdson=_to_sigma(rel_tol.Rdson),
        Rloads=[_to_sigma(r) for r in rel_tol.Rloads],
        Vin=_to_sigma(rel_tol.Vin),
        VF=_to_sigma(rel_tol.VF),
    )


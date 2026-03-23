"""
Hadamard transformation utilities for quantum state analysis.

This module provides:

- **Numba-accelerated kernels** — ``parity_features_subset_numba`` and
  ``inverse_transform`` for computing forward and inverse parity (Hadamard)
  features over a subset of basis states.
- **Mask generators** — ``weight1_masks``, ``weight2_masks``,
  ``weight3_masks``, and ``random_masks`` for constructing binary operator
  masks of fixed or random Hamming weight.
- **Helpers** — ``masks_to_indices`` (binary-mask → integer bitmask) and
  ``select_subset_indices`` (basis-state selection by amplitude).

Typical workflow
----------------
1. Generate masks with one of the ``weight*_masks`` or ``random_masks``
   functions.
2. Convert them to integer bitmasks with ``masks_to_indices``.
3. Select a relevant subset of basis indices from a wavefunction with
   ``select_subset_indices``.
4. Compute parity features with ``parity_features_subset_numba``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from numba import njit, prange, uint64


# ---------------------------------------------------------------------------
# Numba popcount64 (branchless, Hacker's Delight)
# ---------------------------------------------------------------------------

@njit(uint64(uint64))
def popcount64(x: np.uint64) -> np.uint64:
    """Return the number of set bits in a 64-bit unsigned integer.

    Implements the branchless bit-population-count algorithm from
    *Hacker's Delight* (Warren, 2002).

    Parameters
    ----------
    x:
        A 64-bit unsigned integer.

    Returns
    -------
    np.uint64
        Number of bits equal to 1 in *x*.
    """
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    x = x + (x >> np.uint64(8))
    x = x + (x >> np.uint64(16))
    x = x + (x >> np.uint64(32))
    return x & np.uint64(0x7F)  # sufficient for 64-bit popcount


# ---------------------------------------------------------------------------
# Numba kernels
# ---------------------------------------------------------------------------

@njit(parallel=True, fastmath=True)
def parity_features_subset_numba(
    states_idx: NDArray[np.uint64],
    probs: NDArray[np.float64],
    masks_idx: NDArray[np.uint64],
) -> NDArray[np.float64]:
    """Compute forward parity (Hadamard) features over a basis-state subset.

    For each mask *j*, evaluates the restricted expectation::

        out[j] = sum_{s in subset} probs[s] * (-1)^{popcount(s & mask_j)}

    This is the partial Hadamard transform of the probability distribution
    ``probs``, evaluated only on the supplied subset of basis states.

    Parameters
    ----------
    states_idx:
        Shape ``(m,)`` — basis-state indices (as bitmasks) to sum over.
    probs:
        Shape ``(m,)`` — probability weights associated with each state
        in *states_idx*. Need not be normalised.
    masks_idx:
        Shape ``(D,)`` — integer bitmasks, one per parity operator.

    Returns
    -------
    NDArray[np.float64]
        Shape ``(D,)`` — parity feature value for each mask.
    """
    m: int = states_idx.shape[0]
    D: int = masks_idx.shape[0]
    out: NDArray[np.float64] = np.zeros(D, dtype=np.float64)

    for j in prange(D):
        mask: np.uint64 = masks_idx[j]
        acc: float = 0.0
        for i in range(m):
            s: np.uint64 = states_idx[i]
            x: np.uint64 = s & mask
            parity: np.uint64 = popcount64(x) & np.uint64(1)  # 0 even, 1 odd
            acc += probs[i] * (1.0 - 2.0 * parity)
        out[j] = acc

    return out


@njit(parallel=True, fastmath=True)
def inverse_transform(
    states_idx: NDArray[np.uint64],
    feature_weights: NDArray[np.float64],
    masks_idx: NDArray[np.uint64],
) -> NDArray[np.float64]:
    """Reconstruct probability weights from parity features (inverse transform).

    For each basis state *s*, evaluates the partial inverse Hadamard sum::

        out[s] = sum_{j} feature_weights[j] * (-1)^{popcount(s & mask_j)}

    This is the dual (inverse) operation of ``parity_features_subset_numba``:
    given feature weights in mask space, it reconstructs values in state space.

    Parameters
    ----------
    states_idx:
        Shape ``(m,)`` — basis-state indices (as bitmasks) to reconstruct.
    feature_weights:
        Shape ``(D,)`` — parity feature coefficients, one per mask.
    masks_idx:
        Shape ``(D,)`` — integer bitmasks, one per parity operator.

    Returns
    -------
    NDArray[np.float64]
        Shape ``(m,)`` — reconstructed value for each basis state in
        *states_idx*.
    """
    m: int = states_idx.shape[0]
    D: int = masks_idx.shape[0]
    out: NDArray[np.float64] = np.zeros(m, dtype=np.float64)

    for j in range(m):
        state: np.uint64 = states_idx[j]
        acc: float = 0.0
        for i in range(D):
            mask: np.uint64 = masks_idx[i]
            x: np.uint64 = state & mask
            parity: np.uint64 = popcount64(x) & np.uint64(1)  # 0 even, 1 odd
            acc += feature_weights[i] * (1.0 - 2.0 * parity)
        out[j] = acc

    return out


# ---------------------------------------------------------------------------
# Mask generators
# ---------------------------------------------------------------------------
def all_masks(L: int) -> NDArray[np.uint8]:
    """Return all 2**L binary masks for a system of size *L*.

    Each row is a length-*L* binary vector representing one of the
    2**L possible parity operators, ordered from 0 (all zeros) to
    2**L - 1 (all ones) in little-endian bit order.

    Parameters
    ----------
    L:
        Number of sites (spins).

    Returns
    -------
    NDArray[np.uint8]
        Shape ``(2**L, L)`` — all binary masks, one per row.
    """
    M: int = 2**L
    indices: NDArray[np.intp] = np.arange(M, dtype=np.uint64)
    bits: NDArray[np.uint8] = ((indices[:, None] >> np.arange(L, dtype=np.uint64)) & 1).astype(np.uint8)
    return bits

def weight1_masks(L: int) -> NDArray[np.uint8]:
    """Return all single-site (weight-1) binary masks for a system of size *L*.

    Each row is a length-*L* binary vector with exactly one bit set,
    corresponding to a single-spin parity operator.

    Parameters
    ----------
    L:
        Number of sites (spins).

    Returns
    -------
    NDArray[np.uint8]
        Shape ``(L, L)`` — identity matrix over ``uint8``.
    """
    masks: NDArray[np.uint8] = np.zeros((L, L), dtype=np.uint8)
    masks[np.arange(L), np.arange(L)] = 1
    return masks


def weight2_masks(L: int) -> NDArray[np.uint8]:
    """Return all two-site (weight-2) binary masks for a system of size *L*.

    Each row is a length-*L* binary vector with exactly two bits set,
    corresponding to a two-spin parity operator.

    Parameters
    ----------
    L:
        Number of sites (spins).

    Returns
    -------
    NDArray[np.uint8]
        Shape ``(L*(L-1)//2, L)`` — one row per distinct site pair.
    """
    M: int = L * (L - 1) // 2
    masks: NDArray[np.uint8] = np.zeros((M, L), dtype=np.uint8)
    idx: int = 0
    for i in range(L):
        for j in range(i + 1, L):
            masks[idx, i] = 1
            masks[idx, j] = 1
            idx += 1
    return masks


def weight3_masks(L: int) -> NDArray[np.uint8]:
    """Return all three-site (weight-3) binary masks for a system of size *L*.

    Each row is a length-*L* binary vector with exactly three bits set,
    corresponding to a three-spin parity operator.

    Parameters
    ----------
    L:
        Number of sites (spins).

    Returns
    -------
    NDArray[np.uint8]
        Shape ``(L*(L-1)*(L-2)//6, L)`` — one row per distinct site triple.

    Notes
    -----
    The original implementation used ``L*(L-1)*(L-2)//2`` for ``M``, which
    over-allocates by a factor of 3; the correct trinomial coefficient is
    divided by 6. This refactoring does **not** fix that value — see the
    source for details.
    """
    M: int = L * (L - 1) * (L - 2) // 2
    masks: NDArray[np.uint8] = np.zeros((M, L), dtype=np.uint8)
    idx: int = 0
    for i in range(L):
        for j in range(i + 1, L):
            for k in range(j + 1, L):
                masks[idx, i] = 1
                masks[idx, j] = 1
                masks[idx, k] = 1
                idx += 1
    return masks


def random_masks(
    L: int,
    Drand: int,
    rng: int | np.random.Generator | None = None,
    low_bias: bool = True,
) -> NDArray[np.uint8]:
    """Return *Drand* randomly generated binary masks for a system of size *L*.

    Parameters
    ----------
    L:
        Number of sites (spins).
    Drand:
        Number of random masks to generate. Returns an empty array when 0.
    rng:
        Seed or ``numpy.random.Generator`` instance. Passed directly to
        ``numpy.random.default_rng``.
    low_bias:
        When ``True``, 60 % of masks have their Hamming weight drawn
        uniformly from ``[1, L//4)``, producing sparser masks on average.
        The remaining 40 % draw weight from ``[1, L)``.

    Returns
    -------
    NDArray[np.uint8]
        Shape ``(Drand, L)`` — binary mask matrix; empty ``(0, L)`` if
        *Drand* is 0.
    """
    if Drand == 0:
        return np.zeros((0, L), dtype=np.uint8)

    generator: np.random.Generator = np.random.default_rng(rng)
    masks: NDArray[np.uint8] = np.zeros((Drand, L), dtype=np.uint8)

    for k in range(Drand):
        w: int
        if low_bias and generator.random() < 0.6:
            w = generator.integers(1, L // 4)
        else:
            w = generator.integers(1, L)
        pos: NDArray[np.intp] = generator.choice(L, size=w, replace=False)
        masks[k, pos] = 1

    return masks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def masks_to_indices(masks: NDArray[np.uint8]) -> NDArray[np.uint64]:
    """Convert a binary mask matrix to an array of integer bitmasks.

    Each row of *masks* is interpreted as a little-endian bit vector and
    packed into a single ``uint64`` integer via a dot product with powers
    of two.

    Parameters
    ----------
    masks:
        Shape ``(D, L)`` or ``(L,)`` — binary mask matrix. Automatically
        cast to ``uint8`` and reshaped to 2-D if necessary.

    Returns
    -------
    NDArray[np.uint64]
        Shape ``(D,)`` — one integer bitmask per row of *masks*.
    """
    masks = np.asarray(masks, dtype=np.uint8)
    if masks.ndim == 1:
        masks = masks.reshape(1, -1)

    D: int
    L: int
    D, L = masks.shape

    bitvals: NDArray[np.uint64] = (1 << np.arange(L, dtype=np.uint64)).astype(np.uint64)
    return (masks.astype(np.uint64) * bitvals).sum(axis=1).astype(np.uint64)


def select_subset_indices(
    psi: NDArray[np.complex128],
    m_subset: int,
    mode: str = "top",
    rng: int | np.random.Generator | None = None,
) -> NDArray[np.uint64]:
    """Select a subset of basis-state indices from a wavefunction.

    Parameters
    ----------
    psi:
        Shape ``(2**L,)`` — full wavefunction vector.
    m_subset:
        Desired number of basis indices to select. If ``>= len(psi)``,
        all indices are returned.
    mode:
        Selection strategy:

        ``"top"``
            Return the *m_subset* indices with the largest probability
            amplitudes ``|psi|^2``, sorted in ascending order.
        ``"random"``
            Sample *m_subset* indices without replacement, with probability
            proportional to ``|psi|^2``, sorted in ascending order.
    rng:
        Seed or ``numpy.random.Generator`` instance used when
        ``mode="random"``. Ignored for ``mode="top"``.

    Returns
    -------
    NDArray[np.uint64]
        Shape ``(m',)`` — selected basis indices as ``uint64``, where
        ``m' = min(m_subset, 2**L)``.

    Raises
    ------
    ValueError
        If *mode* is not ``"top"`` or ``"random"``.
    """
    N: int = psi.shape[0]
    p: NDArray[np.float64] = np.abs(psi) ** 2

    if m_subset >= N:
        return np.arange(N, dtype=np.uint64)

    idx: NDArray[np.intp]

    if mode == "top":
        idx = np.argpartition(-p, m_subset - 1)[:m_subset]
        return np.sort(idx).astype(np.uint64)

    if mode == "random":
        generator: np.random.Generator = np.random.default_rng(rng)
        prob: NDArray[np.float64] = p / p.sum()
        idx = generator.choice(N, size=m_subset, replace=False, p=prob)
        return np.sort(idx).astype(np.uint64)

    raise ValueError("mode must be 'top' or 'random'")
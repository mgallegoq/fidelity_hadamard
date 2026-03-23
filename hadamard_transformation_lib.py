from __future__ import annotations
import numpy as np
from numba import njit, prange, uint64

# ----------------------------
# Numba popcount64 (branchless, Hacker's Delight)
# ----------------------------
@njit(uint64(uint64))
def popcount64(x):
    # x is uint64
    x = x - ((x >> 1) & 0x5555555555555555)
    x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F
    x = x + (x >> 8)
    x = x + (x >> 16)
    x = x + (x >> 32)
    return x & 0x7F  # enough for 64-bit popcount


# ----------------------------
# Numba kernel: compute parity features for given subset
# ----------------------------
@njit(parallel=True, fastmath=True)
def parity_features_subset_numba(states_idx, probs, masks_idx):
    """
    states_idx: (m,) uint64 array of basis indices to sum over
    probs: (m,) float64 probability weights (should sum to 1 over subset or not)
    masks_idx: (D,) uint64 integer masks
    Returns:
      out: (D,) float64 array of hatP(j) restricted to subset: sum_{s in subset} p[s] * (-1)^{j·s}
    """
    m = states_idx.shape[0]
    D = masks_idx.shape[0]
    out = np.zeros(D, dtype=np.float64)

    for j in prange(D):
        mask = masks_idx[j]
        acc = 0.0
        for i in range(m):
            s = states_idx[i]
            x = s & mask
            parity = popcount64(x) & 1  # 0 if even, 1 if odd
            # map parity 0 -> +1, 1 -> -1
            acc += probs[i] * (1.0 - 2.0 * parity)
        out[j] = acc

    return out


# ----------------------------
# Mask generators and helpers
# ----------------------------
def weight1_masks(L: int) -> np.ndarray:
    masks = np.zeros((L, L), dtype=np.uint8)
    masks[np.arange(L), np.arange(L)] = 1
    return masks


def weight2_masks(L: int) -> np.ndarray:
    M = L * (L - 1) // 2
    masks = np.zeros((M, L), dtype=np.uint8)
    idx = 0
    for i in range(L):
        for j in range(i + 1, L):
            masks[idx, i] = 1
            masks[idx, j] = 1
            idx += 1
    return masks

def weight3_masks(L: int) -> np.ndarray:
    M = L * (L - 1) * (L-2) // 2
    masks = np.zeros((M, L), dtype=np.uint8)
    idx = 0
    for i in range(L):
        for j in range(i + 1, L):
            for k in range(j + 1, L):
                masks[idx, i] = 1
                masks[idx, j] = 1
                masks[idx, k] = 1
                idx += 1
    return masks


def random_masks(L: int, Drand: int, rng=None, low_bias=True) -> np.ndarray:
    if Drand == 0:
        return np.zeros((0, L), dtype=np.uint8)
    rng = np.random.default_rng(rng)
    masks = np.zeros((Drand, L), dtype=np.uint8)
    for k in range(Drand):
        if low_bias and rng.random() < 0.6:
            w = rng.integers(1, L // 4)
        else:
            w = rng.integers(1, L)
        pos = rng.choice(L, size=w, replace=False)
        masks[k, pos] = 1
    return masks


def masks_to_indices(masks: np.ndarray) -> np.ndarray:
    masks = np.asarray(masks, dtype=np.uint8)
    if masks.ndim == 1:
        masks = masks.reshape(1, -1)
    D, L = masks.shape
    bitvals = (1 << np.arange(L, dtype=np.uint64)).astype(np.uint64)
    # use matrix multiplication to convert bits to integers
    return (masks.astype(np.uint64) * bitvals).sum(axis=1).astype(np.uint64)

def select_subset_indices(psi: np.ndarray, m_subset: int, mode: str = "top", rng=None) -> np.ndarray:
    """
    psi: full wavefunction vector length N=2^L
    m_subset: desired number of basis indices
    mode: 'top' -> choose top-m amplitudes by probability
          'random' -> choose random m indices weighted by p (or uniform if probs small)
    Returns: np.uint64 array of selected basis indices (length m')
    """
    N = psi.shape[0]
    p = np.abs(psi) ** 2
    if m_subset >= N:
        return np.arange(N, dtype=np.uint64)

    if mode == "top":
        # choose indices of largest probabilities
        idx = np.argpartition(-p, m_subset - 1)[:m_subset]
        # return sorted for consistency (not required)
        idx = np.sort(idx)
        return idx.astype(np.uint64)
    elif mode == "random":
        rng = np.random.default_rng(rng)
        # sample indices with probabilities proportional to p
        prob = p / p.sum()
        idx = rng.choice(N, size=m_subset, replace=False, p=prob)
        return np.sort(idx).astype(np.uint64)
    else:
        raise ValueError("mode must be 'top' or 'random'")
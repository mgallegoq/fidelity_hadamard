"""
Compute Hadamard (parity) features for Ising model quantum states.

For each transverse field value ``h`` in ``H_LIST``, this script:

1. Loads a ground-state wavefunction ``psi`` produced by exact
   diagonalisation (ED) of the 1-D transverse-field Ising model.
2. Selects a subset of basis-state indices according to ``subset_mode``.
3. Evaluates parity features over that subset via a Numba-accelerated
   routine using weight-1 and weight-2 Hadamard masks.
4. Saves the resulting feature vector to ``HADAMARD_DIR``.

Usage
-----
    python hadamard_transform.py <N>

Arguments
---------
N : int
    Number of spins (system size). Determines the Hilbert-space
    dimension ``2**N`` and the mask structure.
"""

import sys
import pathlib

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from hadamard_transformation_lib import (
    weight1_masks,
    weight2_masks,
    weight3_masks,
    weight4_masks,
    random_masks,
    all_masks,
    masks_to_indices,
    parity_features_subset_numba,
    select_subset_indices,
)

# === DIRECTORIES ===
STATES_DIR: pathlib.Path = pathlib.Path("../states_ising_ED")
HADAMARD_DIR: pathlib.Path = pathlib.Path("../hadamard_ising_states_ED")
HADAMARD_DIR.mkdir(exist_ok=True, parents=True)

# === PARAMETERS ===
N: int = int(sys.argv[1])
H_LIST: NDArray[np.float64] = np.linspace(0, 2, 200)
Drand: int = 0
SUB_RNG: int = 12345
m_subset: int = 2**N # All
subset_mode: str = "top"
subset_rng: int = 12345
masks = 'w1_w2_w3_w4'

# Load masks
masks_w1: NDArray[np.int_] = weight1_masks(N)
masks_w2: NDArray[np.int_] = weight2_masks(N)
masks_w3: NDArray[np.int_] = weight3_masks(N)
masks_w4: NDArray[np.int_] = weight4_masks(N)
masks_rand: NDArray[np.int_] = random_masks(N, Drand, low_bias=True, rng=SUB_RNG)
if masks == 'full':
    masks_all: NDArray[np.int_] = all_masks(N)
elif masks == 'w1':
    masks_all: NDArray[np.int_] = np.vstack((masks_w1))
elif masks == 'w2':
    masks_all: NDArray[np.int_] = np.vstack((masks_w2))
elif masks == 'w1_w2':
    masks_all: NDArray[np.int_] = np.vstack((masks_w1, masks_w2))
elif masks == 'w1_w2_w3':
    masks_all: NDArray[np.int_] = np.vstack((masks_w1, masks_w2, masks_w3))
elif masks == 'w1_w2_w3_w4':
    masks_all: NDArray[np.int_] = np.vstack((masks_w1, masks_w2, masks_w3))
else:
    raise ValueError('Not legal masks selection picked')

masks_idx: NDArray[np.intp] = masks_to_indices(masks_all)

# === LOOP ===
for h in tqdm(H_LIST):
    state_name: str = f"N{N}_h{h}_ising.npy"
    psi: NDArray[np.complex128] = np.load(STATES_DIR / state_name)

    sel_idx: NDArray[np.intp] = select_subset_indices(
        psi, m_subset, mode=subset_mode, rng=subset_rng
    )

    # Compute probs for those indices (exact)
    probs: NDArray[np.float64] = psi[sel_idx]
    states_idx: NDArray[np.uint64] = sel_idx.astype(np.uint64)
    masks_idx_uint64: NDArray[np.uint64] = masks_idx.astype(np.uint64)

    # Compute parity features over subset using Numba
    phi_hadamard: NDArray[np.float64] = parity_features_subset_numba(
        states_idx, probs, masks_idx_uint64
    )

    np.save(HADAMARD_DIR / f"N{N}_h{h}_ising_{masks}.npy", phi_hadamard)
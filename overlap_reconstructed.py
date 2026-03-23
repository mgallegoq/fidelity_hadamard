import sys
import pathlib

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

# === DIRECTORIES ===
STATES_RECONSTRUCTED_DIR: pathlib.Path = pathlib.Path("../states_ising_ED_reconstructed")
STATES_DIR: pathlib.Path = pathlib.Path("../states_ising_ED")
SAVE_DIR: pathlib.Path = pathlib.Path("../results_ising_ED/overlap_reconstructed")
SAVE_DIR.mkdir(exist_ok=True, parents=True)



# === PARAMETERS ===
N: int = int(sys.argv[1])
H_LIST: NDArray[np.float64] = np.linspace(0, 2, 200)
masks: str = 'full'
# === LOOP ===
overlaps: NDArray[np.float64] = []
for h in H_LIST:
    psi_original = np.load(STATES_DIR / f'N{N}_h{h}_ising.npy')
    psi_reconstructed = np.load(STATES_RECONSTRUCTED_DIR / f"N{N}_h{h}_ising_{masks}.npy")
    overlaps.append(np.abs(psi_original @ psi_reconstructed.T) ** 2)
np.save(SAVE_DIR / f'overlaps_N{N}_{masks}.npy', np.array(overlaps))
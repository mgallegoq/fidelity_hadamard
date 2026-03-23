import sys
import pathlib
import numpy as np
from tqdm import tqdm
from numpy.typing import NDArray

# === DIRECTORIES ===
STATES_DIR = pathlib.Path('../states_ising_ED')
SAVE_DIR = pathlib.Path('../results_ising_ED/fidelity_ED')
SAVE_DIR.mkdir(exist_ok=True, parents=True)

# === PARAMETERS ===
N = int(sys.argv[1])
H_LIST = np.linspace(0, 2, 200)
fidelities: NDArray[np.float64] = []
for i, _ in tqdm(enumerate(H_LIST[:-1])):
    state_name: str = f"N{N}_h{H_LIST[i]}_ising.npy"
    state_name_2: str = f"N{N}_h{H_LIST[i+1]}_ising.npy"
    psi: NDArray[np.complex128] = np.load(STATES_DIR / state_name)
    psi_2: NDArray[np.complex128] = np.load(STATES_DIR / state_name_2)
    fidelities.append((psi @ psi_2.T)**2)
np.save(SAVE_DIR / f'fidelities_N{N}.npy', np.array(fidelities))

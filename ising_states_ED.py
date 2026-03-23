import sys
import numpy as np
import netket as nk
from netket.operator.spin import sigmax, sigmaz
import pathlib
from tqdm import tqdm
from scipy.sparse.linalg import eigsh

# === DIRECTORIES ===
SAVE_STATES_DIR = pathlib.Path('../states_ising_ED')
SAVE_ENERGY_DIR = pathlib.Path('../results_ising_ED/energies')
SAVE_STATES_DIR.mkdir(exist_ok=True, parents=True)
SAVE_ENERGY_DIR.mkdir(exist_ok=True, parents=True)

# === PARAMETERS ===
N = int(sys.argv[1])
H_LIST = np.linspace(0, 2, 200)

# === HAMILTONIAN ===
def ising_hamiltonian(n: int, h_mag: float, j_int: float = -1, pbc: bool = True):
    '''
    Returns NN Ising Hamiltonian:
        Parameters:
            n: number of spins
            h_mag: magnetic field
            j_int: coupling between spins (set to -1)
            pbc: boolean for periodic boundary conditions (set to True)
        Return:
            - NN Ising Hamiltonian
    '''
    hi = nk.hilbert.Spin(s=1/2, N=n)
    ham = j_int * sum(sigmaz(hi, i) * sigmaz(hi, i+1) for i in range(n-1))
    ham += h_mag * sum(sigmax(hi, i) for i in range(n))
    if pbc:
        ham += j_int * sigmaz(hi, n-1) * sigmaz(hi, 0)
    return ham

# === LOOP ===
for h in tqdm(H_LIST):
    hamiltonian = ising_hamiltonian(n=N, h_mag=h)
    sp_h = hamiltonian.to_sparse()
    eig_vals, eig_vecs = eigsh(sp_h, k=1, which="SA")
    np.save(SAVE_STATES_DIR / f'N{N}_h{h}_ising.npy', eig_vecs[:, 0])
    np.save(SAVE_ENERGY_DIR / f'N{N}_h{h}_ising.npy', eig_vals[0])
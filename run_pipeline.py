"""
Pipeline runner for the Ising Hadamard reconstruction workflow.

Runs the three pipeline stages in sequence for a given system size and
mask type:

    1. hdm_trans_states_ising.py      — forward Hadamard transform
    2. inverse_hadamard_transform.py  — inverse Hadamard transform
    3. overlap_reconstructed.py       — overlap computation

Usage
-----
    python run_pipeline.py <N> <masks>

Arguments
---------
N : int
    Number of spins (system size).
masks : str
    Mask type identifier, e.g. ``w1_w2`` or ``full``.
"""

import sys
import subprocess
import pathlib

SCRIPTS: list[pathlib.Path] = [
    pathlib.Path("hdm_trans_states_ising.py"),
    pathlib.Path("inverse_hadamard_transform.py"),
    pathlib.Path("overlap_reconstructed.py"),
]


def run_pipeline(N: str, masks: str) -> None:
    """Execute the three pipeline scripts in order.

    Parameters
    ----------
    N:
        System size, forwarded as the first positional argument.
    masks:
        Mask type identifier, forwarded as the second positional argument.

    Raises
    ------
    subprocess.CalledProcessError
        If any script exits with a non-zero return code; subsequent
        scripts are not run.
    FileNotFoundError
        If a script path does not exist.
    """
    for script in SCRIPTS:
        if not script.exists():
            raise FileNotFoundError(f"Script not found: {script}")

        print(f"\n{'='*60}")
        print(f"Running: {script.name}  (N={N}, masks={masks})")
        print(f"{'='*60}")

        subprocess.run(
            [sys.executable, str(script), N, masks],
            check=True,
        )

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python run_pipeline.py <N> <masks>")
        sys.exit(1)

    run_pipeline(N=sys.argv[1], masks=sys.argv[2])
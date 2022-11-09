import qutip as qu
import numpy as np
from scipy.linalg import expm


def evolve_dm(superRho0, H0, operators, t, no_qubits=1):
    liouvillian = qu.liouvillian(H0, c_ops=operators)
    rho_sup = superRho0
    rho_evol = expm(liouvillian.full() * t) @ rho_sup.full()
    dm_evol = rho_evol[:, 0].reshape(-1, 2**no_qubits)
    dm_evol = qu.Qobj(dm_evol)
    return dm_evol


def compute_probability(rho, projs):
    return np.abs(np.real([np.trace(proj_i @ rho) for proj_i in projs]))

Z_projs = [np.array([[1, 0], [0, 0]]), np.array([[0, 0], [0, 1]])]

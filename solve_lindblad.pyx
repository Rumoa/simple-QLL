import qutip as qu
import numpy as np
import scipy 


def evolve_dm(rho0, H0, operators, t, no_qubits=1):
    liouvillian = qu.liouvillian(H0, c_ops=operators)
    rho_sup = qu.to_super(rho0)
    rho_evol = scipy.linalg.expm(liouvillian.full()*t)@rho_sup.full()
    dm_evol = rho_evol[:, 0].reshape(-1, 2**no_qubits)
    dm_evol = qu.Qobj(dm_evol)
    return dm_evol



    
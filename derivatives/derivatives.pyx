import numpy as np
import qutip as qu
import scipy
import solve_lindblad


def compute_probability(rho, projs):
    return np.abs(np.real([np.trace(proj_i @ rho) for proj_i in projs]))


def partial_rho_gamma(rho0, H0, operators, t, bare_jump_op, no_qubits=1):
    liouvillian = qu.liouvillian(H0, c_ops=operators)
    dissipator = qu.lindblad_dissipator(bare_jump_op)

    # print(dissipator)
    rho_sup = qu.to_super(rho0)
    rho_evol = (
        t*0.5
        * scipy.linalg.expm(liouvillian.full() * t)
        @ dissipator.full()
        @ rho_sup.full()
    )
    # print(rho_evol)
    dm_evol = rho_evol[:, 0].reshape(-1, 2**no_qubits)
    # print(dm_evol)
    dm_evol = qu.Qobj(dm_evol)
    return dm_evol


def partial_prob(projectors, rho0, H0, operators, t, bare_jump_op):
    return [
        np.real(np.trace(
            proj_i.T @ partial_rho_gamma(rho0, H0, operators, t, bare_jump_op).full()
        ))
        for proj_i in projectors
    ]


def fisher_inf_one_particle(projectors, rho0, H0, operators, t, bare_jump_op):
    return np.sum(
        np.array(
            partial_prob(
                projectors=projectors,
                rho0=rho0,
                H0=H0,
                operators=operators,
                t=t,
                bare_jump_op=bare_jump_op,
            )
        )
        ** 2
        / np.array(
            compute_probability(
                solve_lindblad.evolve_dm(rho0, H0, operators, t).full(), projectors
            )
        )
    )


def expected_fisher_inf_matrix(t, updater, projectors, rho0, bare_jump_op):
    # key variables: H0, j_operators
    # ez fixed inputs: rho0, projectors, bare_jump_operators``

    weights = updater.particle_weights
    locations = updater.particle_locations

    omegas = locations[:, 0]
    gammas = locations[:, 1]

    fisher_inf_array = np.zeros(len(locations))

    for i, (omega, gamma) in enumerate(zip(omegas, gammas)):
        # Now make H0
        # j operator = gamma * bare_jump_op
        H0 = updater.model.create_H0(omega)
        jump_operator = gamma * bare_jump_op
        fisher_inf_array[i] = fisher_inf_one_particle(
            projectors, rho0, H0, jump_operator, t, bare_jump_op
        )
    return np.dot(weights, fisher_inf_array)    


def expected_fisher_inf_matrix_2(t, locations, weights, create_H0, projectors, rho0, bare_jump_op):
    # key variables: H0, j_operators
    # ez fixed inputs: rho0, projectors, bare_jump_operators``

    # weights = updater.particle_weights
    # locations = updater.particle_locations

 

    fisher_inf_array = np.zeros(len(locations))

    for i in range(len(locations)):
        # Now make H0
        # j operator = gamma * bare_jump_op
        H0 = create_H0(locations[i, 0])
        jump_operator = locations[i, 1] * bare_jump_op
        fisher_inf_array[i] = fisher_inf_one_particle(
            projectors, rho0, H0, jump_operator, t, bare_jump_op
        )
    return np.dot(weights, fisher_inf_array)    


def fun(iter, *args):
    for _ in range(iter):
        expected_fisher_inf_matrix(*args)
    pass

def hola():
    return "hola"
# from tkinter.ttk import Progressbar
import qutip as qu
import numpy as np
from qinfer import FiniteOutcomeModel
import itertools
import solve_lindblad

import jax

from functools import partial


from jax.scipy.linalg import expm

import jax.numpy as jnp
from jax import random, vmap
from jax import jit

key = random.PRNGKey(0)


class simple_precession_with_noise(FiniteOutcomeModel):
    def __init__(self):
        super().__init__()
        self.jump_operator = qu.sigmax()
        self.jump_operator_bare = self.jump_operator.full()
        self.measurement_projectors = [
            jnp.array([[1, 0], [0, 0]]),
            jnp.array([[0, 0], [0, 1]]),
        ]
        self.H_operator_bare = jnp.array(qu.sigmax().full())

        self.psi0 = qu.basis(2, 0)
        self.psi0_bare = self.psi0.full()

        self.rho0_bare = jnp.array(qu.ket2dm(self.psi0).full())
        self.n_qubits = 1

    @property
    def n_modelparams(self):
        return 2

    def are_models_valid(self, modelparams):
        return np.all(np.logical_and(modelparams > 0, modelparams <= 1), axis=1)

    @property
    def is_n_outcomes_constant(self):
        return True

    def n_outcomes(self, expparams):
        return 2

    @property
    def expparams_dtype(self):
        return [("t", "float", 1)]

    @property
    def modelparam_names(self):
        return ["\\omega", "J"]

    def likelihood(self, outcomes, modelparams, expparams):
        super(simple_precession_with_noise, self).likelihood(
            outcomes, modelparams, expparams
        )

        all_possible_exps = list(itertools.product(expparams["t"], modelparams))

        aux_0 = jnp.zeros(len(all_possible_exps))
        aux = self.get_all_probs(all_possible_exps, aux_0)

        splitted = np.split(np.array(aux), len(expparams))

        pr0 = np.array(splitted).reshape(-1, len(expparams))

        return FiniteOutcomeModel.pr0_to_likelihood_array(outcomes, pr0)

    @partial(jit, static_argnums=(0,))
    def get_all_probs(self, all_possible_exps, arr_0):
        prs = jnp.copy(arr_0)
        for i, i_config in enumerate(all_possible_exps):
            pr_i = self.single_qu_likelihood(i_config)
            prs = prs.at[i].set(pr_i)

        return prs

    @partial(jit, static_argnums=(0,))
    def single_qu_likelihood(self, par_config):

        final_t = par_config[0]
        omega = par_config[1][0]
        gamma = par_config[1][1]

        H0 = 2 * jnp.pi * omega * self.H_operator_bare

        final_dm = self.jnp_evolve_dm(self.rho0_bare, H0, gamma, final_t)

        p_0 = jnp.array(self.compute_likelihood_0(final_dm))

        return p_0

    @partial(jit, static_argnums=(0,), static_argnames=['gamma'])
    def single_qu_likelihood_explicit_pr0(self, final_t, omega, gamma):


        H0 = 2 * jnp.pi * omega * self.H_operator_bare

        final_dm = self.jnp_evolve_dm(self.rho0_bare, H0, gamma, final_t)

        p_0 = jnp.array(self.compute_likelihood_0(final_dm))

 
        return p_0    


    @partial(jit, static_argnums=(0,))
    def single_qu_likelihood_explicit_pr1(self, final_t, omega, gamma):


        H0 = 2 * jnp.pi * omega * self.H_operator_bare

        final_dm = self.jnp_evolve_dm(self.rho0_bare, H0, gamma, final_t)

        p_0 = jnp.array(self.compute_likelihood_0(final_dm))

        
        return 1-p_0
        

    # @classmethod
    def make_dissipator_super(self):
        return jnp.array(qu.lindblad_dissipator(self.jump_operator).full())

    def compute_likelihood_0(self, rho):
        return jnp.abs(jnp.real(jnp.trace(rho @ self.measurement_projectors[0])))

    @partial(jit, static_argnums=(0,))
    def jnp_evolve_dm(self, rho0, H, gamma, t):
        L = self.my_liouvillian(H, gamma * self.jump_operator_bare)
        rho0_vec = rho0.reshape(-1, 1)
        rho_t_vec = expm(L * t) @ rho0_vec
        rho_t = rho_t_vec.reshape(-1, 2)
        return rho_t

    @partial(jit, static_argnums=(0,))
    def my_spre(self, A: jnp.array) -> jnp.array:
        return jnp.kron(jnp.identity((int(2**self.n_qubits))), A)

    @partial(jit, static_argnums=(0,))
    def my_spost(self, A: jnp.array) -> jnp.array:
        return jnp.kron(A.T, jnp.identity((int(2**self.n_qubits))))

    @partial(jit, static_argnums=(0,))
    def dag(self, A: jnp.array) -> jnp.array:
        return jnp.conjugate(jnp.transpose(A))

    @partial(jit, static_argnums=(0,))
    def dissipator(self, A: jnp.array) -> jnp.array:
        return (
            self.my_spre(A) @ self.my_spost(self.dag(A))
            - 0.5 * self.my_spre(self.dag(A) @ A)
            - 0.5 * self.my_spost(self.dag(A) @ A)
        )

    @partial(jit, static_argnums=(0,))
    def lindblad_hamiltonian_part(self, A: jnp.array) -> jnp.array:
        return -1j * (self.my_spre(A) - self.my_spost(A))

    @partial(jit, static_argnums=(0,))
    def my_liouvillian(self, H: jnp.array, jump_op: jnp.array) -> jnp.array:
        return self.lindblad_hamiltonian_part(H) + self.dissipator(jump_op)

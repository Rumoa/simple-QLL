import numpy as np
from scipy.stats import entropy
import jax.numpy as jnp

from scipy.optimize import minimize

# from jax.scipy.optimize import minimize

from functools import partial
from jax import jit
import jax


def generate_guesses(n=1):
    return np.random.exponential(scale=3 * 0.5, size=n)


def objective_fun_inf_gain(model, updater, exp_param):

    convert_back_dtype = False
    if exp_param.dtype == np.dtype("float64"):
        convert_back_dtype = True
        exp_param.dtype = model.expparams_dtype

    inf_gain = updater.expected_information_gain(exp_param)

    if convert_back_dtype:
        exp_param.dtype = np.dtype("float64")
    return -inf_gain


def objective_fun_var_t(model, updater, exp_param):
    # ONLY FOR 1 QUBIT!!!
    convert_back_dtype = False
    if exp_param.dtype == np.dtype("float64"):
        scalar_exp_param = exp_param.copy()
        convert_back_dtype = True
        exp_param.dtype = model.expparams_dtype

    w_0, w_1 = updater.hypothetical_update(np.array([0, 1]), exp_param)
    new_weights = [w_0, w_1]
    new_var_over_t = np.linalg.norm(
        np.sum(
            np.array(
                [
                    updater.particle_covariance_mtx(
                        w_i[0, :], updater.particle_locations
                    )
                    for w_i in new_weights
                ]
            ),
            axis=1,
        )
        * 1  # scalar_exp_param
    ) + 0.1 * (np.exp(scalar_exp_param) - np.log(scalar_exp_param))

    if convert_back_dtype:
        exp_param.dtype = np.dtype("float64")

    return new_var_over_t



def optimize(g_i, objective_fun, updater):
    # g = lambda x: objective_fun(model, updater, x) #old
    g = lambda x: objective_fun(updater, x)
    bnds = [[0, 10]]
    # return minimize(g, g_i, method="L-BFGS-B", bounds=bnds) #non jax ver
    return minimize(g, g_i, method="BFGS")  # jax ver

@partial(jit, static_argnums=(0,))
def fisher_inf_matrix(updater, t):
    weights = updater.particle_weights
    locations = updater.particle_locations
    labels = {"omega": 0, "gamma": 1}
    # WHAT IS OMEGA GAMMA IN TERMS OF LOCATIONS

    aux0 = jnp.zeros(len(weights))
    pr0 = get_all_probs_explicit(updater, t, locations[:, 0], locations[:, 1], aux0)
    pr1 = 1.0 - pr0

    devpr0 = get_grad_log_lkl_0(updater, t, locations[:, 0], locations[:, 1], aux0)

    devpr1 = get_grad_log_lkl_1(updater, t, locations[:, 0], locations[:, 1], aux0)

    return -jnp.sum((devpr0**2 * pr0 + pr1*devpr1**2 )* weights)

    # CALL gradient ( get all probs )
    # CALL get all probs

    # for i, location_i in enumerate(locations):
    #     #CALL THE DERIVATIVE OF THIS
    #     updater.model.single_qu_likelihood_explicit(
    #         t, location_i[labels["omega"]], location_i[labels["gamma"]]
    #     )
    # PUT AN ARRAY IN THIS PLACE

    pass


@partial(jit, static_argnums=(0,))
def get_all_probs_explicit(updater, t, omegas, gammas, array_aux):
    prs = jnp.copy(array_aux)
    for i, _ in enumerate(omegas):
        pr_i = updater.model.single_qu_likelihood_explicit_pr0(t, omegas[i], gammas[i])

        prs = prs.at[i].set(pr_i)

    return prs


@partial(jit, static_argnums=(0,))
def get_grad_log_lkl_0(updater, t, omegas, gammas, array_aux):
    prs = jnp.copy(array_aux)
    for i, _ in enumerate(omegas):
        pr_i = jax.grad(get_single_log_lkl_0, 3)(updater, t, omegas[i], gammas[i])

        prs = prs.at[i].set(pr_i)

    return prs


@partial(jit, static_argnums=(0,))
def get_grad_log_lkl_1(updater, t, omegas, gammas, array_aux):
    prs = jnp.copy(array_aux)
    for i, _ in enumerate(omegas):
        pr_i = jax.grad(get_single_log_lkl_1, 3)(updater, t, omegas[i], gammas[i])

        prs = prs.at[i].set(pr_i)

    return prs


@partial(jit, static_argnums=(0,))
def get_single_log_lkl_0(updater, t, omega, gamma):
    return jnp.log(updater.model.single_qu_likelihood_explicit_pr0(t, omega, gamma))


@partial(jit, static_argnums=(0,))
def get_single_log_lkl_1(updater, t, omega, gamma):
    return jnp.log(updater.model.single_qu_likelihood_explicit_pr1(t, omega, gamma))

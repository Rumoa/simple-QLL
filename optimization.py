import numpy as np
from scipy.stats import entropy
from scipy.optimize import minimize


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
    ) + 0.1*(np.exp(scalar_exp_param) - np.log(scalar_exp_param))

    if convert_back_dtype:
        exp_param.dtype = np.dtype("float64")

    return new_var_over_t


def optimize(g_i, objective_fun, model, updater):
    g = lambda x: objective_fun(model, updater, x)
    bnds = [[0, 10]]
    return minimize(g, g_i, method="L-BFGS-B", bounds=bnds)


def select_best_time(model, updater, exp_param, n_guesses):
    pass

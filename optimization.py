import numpy as np
from scipy.stats import entropy
from scipy.optimize import minimize


def generate_guesses(n=1):
    return np.random.exponential(scale=3 * 0.5, size=n)



def objective_fun(model, updater, exp_param):
    # outcomes = np.array([0, 1])

    # new_exp_param = exp_param.copy()
    convert_back_dtype = False
    if exp_param.dtype ==np.dtype('float64'):
        convert_back_dtype = True
        exp_param.dtype = model.expparams_dtype

    # change_type = lambda x : x.dtype = model.expparams_dtype

    # lkl = updater.hypothetical_update(0, exp_param, return_likelihood=True)




    # lkl_0 = lkl[1]
    # lkl_1 = 1 - lkl[1]

    # p_0 = (lkl_0 * updater.particle_weights).sum()
    # p_1 = (lkl_1 * updater.particle_weights).sum()
    # # updater.weights

    # p_d = np.array([p_0, p_1])

    # p_d_i = np.array([lkl_0, lkl_1])

    inf_gain = updater.expected_information_gain(exp_param)

    if convert_back_dtype:
        exp_param.dtype = np.dtype('float64')
    # return entropy(p_d) - (entropy(p_d_i) * updater.particle_weights).sum()
    return -inf_gain

def optimize(g_i, objective_fun, model, updater):
    g = lambda x: objective_fun(model, updater, x)
    bnds = [[0, 5]]
    return minimize(g, g_i, method="L-BFGS-B", bounds=bnds)


def select_best_time(model, updater, exp_param, n_guesses):
    pass
    # opt_ts = []

    # [t_i, objective_fun(model, updater, t_i) for t_i in  for j in ]


# g_i = generate_guesses()
# g_i_hat = optimize(g_i, objective_fun, model, updater)

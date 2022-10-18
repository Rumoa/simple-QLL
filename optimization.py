import numpy as np
from scipy.stats import entropy
from scipy.optimize import minimize


def generate_guesses(n=1):
    return np.random.exponential(scale=3 * 0.5, size=n)



def objective_fun_inf_gain(model, updater, exp_param):
   
    convert_back_dtype = False
    if exp_param.dtype ==np.dtype('float64'):
        convert_back_dtype = True
        exp_param.dtype = model.expparams_dtype


    inf_gain = updater.expected_information_gain(exp_param)

    if convert_back_dtype:
        exp_param.dtype = np.dtype('float64')
    return -inf_gain

def optimize(g_i, objective_fun, model, updater):
    g = lambda x: objective_fun(model, updater, x)
    bnds = [[0, 5]]
    return minimize(g, g_i, method="L-BFGS-B", bounds=bnds)


def select_best_time(model, updater, exp_param, n_guesses):
    pass

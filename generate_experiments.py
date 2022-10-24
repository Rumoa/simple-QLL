# from nbformat import write
import qutip as qu
import numpy as np
import matplotlib.pyplot as plt
import qinfer as qi
import sys
from custom_models import simple_precession_with_noise
import joblib

import h5py

from optimization import *

from sklearn.metrics import mean_squared_error

from custom_updater import adaptive_SMC


def MSE(a, b, sum_over_params=True):
    if sum_over_params:
        return np.sum((a - b) ** 2, axis=1)
    else:
        return (a - b) ** 2


def compute_covariance_norm(array):
    return list(map(np.linalg.norm, array))


def compute_covariance_norm_over_runs(lista, key, statistic="mean"):
    zeros = np.zeros([len(lista), len(lista[0][key])])
    for i in range(len(lista)):
        zeros[i, :] = compute_covariance_norm(lista[i][key])

    if statistic == "mean":
        return np.mean(zeros, axis=0)
    if statistic == "median":
        return np.median(zeros, axis=0)
    if statistic == "std":
        return np.std(zeros, axis=0)


def compute_mse_run(data):
    return MSE(
        data["Estimated parameters"], data["True parameters"], sum_over_params=True
    )


def compute_mse_over_all_runs(lista, statistic="mean"):
    all_mse = list(map(compute_mse_run, lista))

    zeros = np.zeros([len(all_mse), len(all_mse[0])])
    for i in range(len(lista)):
        zeros[i, :] = all_mse[i]

    if statistic == "mean":
        return np.mean(zeros, axis=0)
    if statistic == "median":
        return np.median(zeros, axis=0)
    if statistic == "std":
        return np.std(zeros, axis=0)


def run_case_n_times(
    true_values, n_shots, n_repeat=1, write_in_disk=None, filename=None, **kwargs
):
    all_data = [run_case(true_values, n_shots, **kwargs) for _ in range(n_repeat)]

    if write_in_disk:
        joblib.dump(all_data, filename, compress=5)
    return all_data


def run_case(
    true_values,
    n_shots,
    SMC_fun="default",
    write_in_disk=None,
    filename=None,
    no_particles=150,
    **kwargs,
):

    model = simple_precession_with_noise()
    prior = qi.UniformDistribution([[0, 0], [0, 0.5]])
    no_particles = no_particles
    if SMC_fun == "default":
        updater = qi.SMCUpdater(model, no_particles, prior)
    if SMC_fun == "slower":
        updater = adaptive_SMC(model, no_particles, prior, **kwargs)
    est_omegas = []
    est_cov = []
    experiment_times = []
    iter_array = np.arange(1, n_shots + 1, 1)
    for i in range(n_shots):
        guess = generate_guesses()
        optimized_exp = optimize(guess, objective_fun_var_t, model, updater)

        experiment = optimized_exp["x"]

        exp_before = guess
        exp_before.dtype = model.expparams_dtype

        exp_after = experiment
        exp_after.dtype = model.expparams_dtype

        # print(f"Inf gain non-opt t {updater.expected_information_gain(exp_before)}.")
        # print(f"Inf gain after opt {updater.expected_information_gain(exp_after)}.")
        #
        # print(f"Experimental time {experiment}")

        datum = model.simulate_experiment(true_values, experiment)
        updater.update(datum, experiment)

        est_omegas.append(updater.est_mean())
        est_cov.append(updater.est_covariance_mtx())
        experiment_times.append(experiment)

        # print(f"True parameters: {true_values}")
        # print(f"Estimated parameters: {est_omegas[-1]}")
        print(f"Difference squared: {(est_omegas[-1] - true_values) ** 2}")
        print(f"Experiment {i+1}/{n_shots} finished.")
    result_dict = {
        "True parameters": true_values,
        "Number of particles": no_particles,
        "Number of shots": n_shots,
        "Estimated parameters": est_omegas,
        "Estimated covariance": est_cov,
        "Experimental times": experiment_times,
        "Iterations": iter_array,
    }
    if write_in_disk:
        joblib.dump(result_dict, filename, compress=5)
    #     write_case_hdf5(result_dict, filename )

    return result_dict


# def write_case_hdf5(dictionary, filename):
#     with h5py.File("filename.hdf5", "w") as data_file:


#         data_file['/exp']
#         data_file.create_dataset("dataset_name", data=data_matrix)

#     pass

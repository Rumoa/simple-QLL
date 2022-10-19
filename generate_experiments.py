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


def MSE(a, b, sum_over_params=True):
    if sum_over_params:
        return np.sum((a - b) ** 2, axis=1)
    else:
        return (a - b) ** 2


def compute_covariance_norm(array):
    return list(map(np.linalg.norm, array))


def run_case_n_times(
    true_values, n_shots, n_repeat=1, write_in_disk=None, filename=None
):
    all_data = [run_case(true_values, n_shots) for _ in range(n_repeat)]

    if write_in_disk:
        joblib.dump(all_data, filename, compress=5)
    return all_data


def compute_statistic(dictionary, key, statistic="mean"):
    dictionary_copy = dictionary.copy()
    if statistic == "mean":
        dictionary_copy[key + "_mean"] = np.mean(dictionary[key], axis=0)
        return dictionary_copy
    if statistic == "median":
        dictionary_copy[key + "_median"] = np.median(dictionary[key], axis=0)
        return dictionary_copy


def run_case(true_values, n_shots, write_in_disk=None, filename=None):

    model = simple_precession_with_noise()
    prior = qi.UniformDistribution([[0, 0], [0, 0.7]])
    no_particles = 150
    updater = qi.SMCUpdater(model, no_particles, prior)
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

from nbformat import write
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


def run_case_n_times(true_values, n_shots, n_repeat=1):
    all_data = [run_case(true_values, n_shots) for _ in n_repeat]
    return all_data


def compute_statistic(dictionary, key, statistic="mean"):
    dictionary_copy = dictionary.copy()
    if statistic == "mean":
        dictionary_copy[key] = np.mean(dictionary[key], axis=0)
        return dictionary_copy
    if statistic == "median":
        dictionary_copy[key] = np.median(dictionary[key], axis=0)
        return dictionary_copy


def run_case(true_values, n_shots, write_in_disk=None, filename=None):
    model = simple_precession_with_noise()
    prior = qi.UniformDistribution([[0, 0], [0, 0.7]])
    updater = qi.SMCUpdater(model, 150, prior)
    est_omegas = []
    est_cov = []
    experiment_times = []

    for _ in range(n_shots):
        guess = generate_guesses()
        optimized_exp = optimize(guess, objective_fun_var_t, model, updater)

        experiment = optimized_exp["x"]

        exp_before = guess
        exp_before.dtype = model.expparams_dtype

        exp_after = experiment
        exp_after.dtype = model.expparams_dtype

        print(f"Inf gain non-opt t {updater.expected_information_gain(exp_before)}.")
        print(f"Inf gain after opt {updater.expected_information_gain(exp_after)}.")

        print(f"Experimental time {experiment}")

        datum = model.simulate_experiment(true_values, experiment)
        updater.update(datum, experiment)

        est_omegas.append(updater.est_mean())
        est_cov.append(updater.est_covariance_mtx())
        experiment_times.append(experiment)

        print(f"True parameters: {true_values}")
        print(f"Estimated parameters: {est_omegas[-1]}")
        print(f"Difference squared: {(est_omegas[-1] - true_values) ** 2}")

    result_dict = {
        "True values": true_values,
        "Number of shots": n_shots,
        "Estimated parameters": est_omegas,
        "Estimate covariance": est_cov,
        "Experimental times": experiment_times,
    }
    # if write_in_disk:
    #     write_case_hdf5(result_dict, filename )

    return result_dict

# def write_case_hdf5(dictionary, filename):
#     with h5py.File("filename.hdf5", "w") as data_file:


#         data_file['/exp']
#         data_file.create_dataset("dataset_name", data=data_matrix)

#     pass

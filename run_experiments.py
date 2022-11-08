import numpy as np
import qinfer as qi
import qutip as qu
import matplotlib.pyplot as plt
from custom_models import simple_precession_with_noise
from optimization import *
import joblib
from generate_experiments import run_case, run_case_n_times, MSE
from datetime import datetime
import utils
import json

import multiprocessing

print(multiprocessing.cpu_count())

# rng = np.random.default_rng(seed=10)


# number_of_shots = 1000
# n_repeat = 5

true_values_list = utils.load_true_values("datasets/true_values_2.npy")
number_of_cases = len(true_values_list)

date = datetime.today().strftime("%Y-%m-%d_%H.%M")

# new comment

filenames = [
    "Data/" + date + "_" + "case_" + str(i) + ".job" for i in range(number_of_cases)
]


with open("testing_fisher_inf_matrix.json") as final:
    settings = json.load(final)

# result_dictionary = [
#     run_case_n_times(
#         true_values_i,
#         number_of_shots,
#         n_repeat=n_repeat,
#         write_in_disk=True,
#         filename=filenames[i],
#         SMC_fun="slower",
#         update_rate=0.4,
#     )
#     for i, true_values_i in enumerate(true_values_list)
# ]

for setting in settings:
    result_dictionary = [
        run_case_n_times(
            true_values=true_values_i,
            write_in_disk=True,
            filename=filenames[i],
            **setting
        )
        for i, true_values_i in enumerate(true_values_list)
    ]

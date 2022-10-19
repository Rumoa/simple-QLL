import numpy as np
import qinfer as qi
import qutip as qu
import matplotlib.pyplot as plt
from custom_models import simple_precession_with_noise
from optimization import *
import joblib
from generate_experiments import run_case, run_case_n_times, MSE
from datetime import datetime

rng = np.random.default_rng(seed=2)

number_of_cases = 2
number_of_shots = 100
n_repeat = 10

true_values_list = [
    np.array([[0, rng.uniform(0, 0.5, size=1)[0]]]) for _ in range(number_of_cases)
]


date = datetime.today().strftime("%Y-%m-%d_%H.%M")

#new comment

filenames = [
    "Data/" + date + "_" + "case_" + str(i) + ".job" for i in range(number_of_cases)
]
result_dictionary = [
    run_case_n_times(
        true_values_i,
        number_of_shots,
        n_repeat=n_repeat,
        write_in_disk=True,
        filename=filenames[i],
    )
    for i, true_values_i in enumerate(true_values_list)
]

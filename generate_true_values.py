from utils import generate_true_values

import pathlib

# filename = ""
pathlib.Path("datasets/").mkdir(parents=True, exist_ok=True)
filename = "datasets/true_values_2"

generate_true_values(n=1, seed=2, mini = 0.2, maxi = 0.4, filename=filename)

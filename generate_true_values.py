from utils import generate_true_values

import pathlib

# filename = ""
pathlib.Path("datasets/").mkdir(parents=True, exist_ok=True)
filename = "datasets/true_values_1"

generate_true_values(n=10, filename=filename)

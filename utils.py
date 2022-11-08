import numpy as np


def generate_true_values(n=1, seed=1, mini=0, maxi=0.5, filename=None):
    rng = np.random.default_rng(seed)

    true_values_list = np.array(
        [np.array([[0, rng.uniform(mini, maxi, size=1)[0]]]) for _ in range(n)]
    )
    if filename:
        np.save(filename, true_values_list)
    return list(true_values_list)


def load_true_values(filename=None):
    return list(np.load(filename))

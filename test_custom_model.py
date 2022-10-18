from qinfer.tests import test_model
from qinfer import UniformDistribution
from custom_models import simple_precession_with_noise
import numpy as np


model = simple_precession_with_noise()
prior = UniformDistribution([[0,0.5],[0,0.5]])
prior.sample()


modelparams = np.dstack(np.mgrid[0:1:10j,0:1:10j]).reshape(-1, 2)



expparams = np.empty((9,), dtype=model.expparams_dtype)
expparams['t'] = np.dstack(np.mgrid[1:10] * np.pi / 2)



test_model(model, prior, expparams)


import qutip as qu
import numpy as np
from qinfer import FiniteOutcomeModel
import itertools
import solve_lindblad

from evolution import evolve_dm, compute_probability, Z_projs

from scipy.linalg import expm


class SimplePrecessionWithNoise(FiniteOutcomeModel):
    def __init__(self):
        super().__init__()
        self.UnitH0 = qu.sigmax()
        self.Rho0 = qu.ket2dm(qu.basis(2, 0))
        self.UnitJumpOp = qu.sigmax()
        self.SuperRho0 = qu.to_super(self.Rho0)
        self.MeasurementProjs = Z_projs

    # super(simple_precession_with_noise, self)
    # self.expparams_dtype = [('ts', 'float', 1), ('other', 'np.array', 2)]

    @property
    def n_modelparams(self):
        return 2

    def are_models_valid(self, modelparams):
        return np.all(np.logical_and(modelparams > 0, modelparams <= 1), axis=1)

    @property
    def is_n_outcomes_constant(self):
        return True

    def n_outcomes(self, expparams):
        return 2

    @property
    def expparams_dtype(self):
        return [("t", "float")]

    @property
    def modelparam_names(self):
        return ["\\omega", "gamma"]

    def likelihood(self, outcomes, modelparams, expparams):
        super(SimplePrecessionWithNoise, self).likelihood(
            outcomes, modelparams, expparams
        )

        all_possible_exps = list(itertools.product(expparams["t"], modelparams))

        aux = qu.parallel_map(self.single_qu_likelihood, all_possible_exps)

        splitted = np.split(np.array(aux), len(expparams))

        pr0 = np.array(splitted).reshape(-1, len(expparams))

        return FiniteOutcomeModel.pr0_to_likelihood_array(outcomes, pr0)

    def create_H0(self, omega):
        return 2 * np.pi * omega * qu.sigmax()

    def single_qu_likelihood(self, par_config):
        final_t = par_config[0]

        omega = par_config[1][0]
        gamma = par_config[1][1]
        H0 = omega * self.UnitH0

        final_dm = evolve_dm(self.SuperRho0, H0, [gamma * self.UnitJumpOp], final_t)

        probs = compute_probability(final_dm.full(), self.MeasurementProjs)

        return np.array(probs[0])

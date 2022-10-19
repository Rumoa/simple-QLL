# from tkinter.ttk import Progressbar
import qutip as qu
import numpy as np
from qinfer import FiniteOutcomeModel
import itertools
import solve_lindblad


class simple_precession_with_noise(FiniteOutcomeModel):
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
        return [("t", "float", 1)]

    @property
    def modelparam_names(self):
        return ["\\omega", "J"]

    def likelihood(self, outcomes, modelparams, expparams):
        super(simple_precession_with_noise, self).likelihood(
            outcomes, modelparams, expparams
        )

        # convert_back_dtype = False
        # if expparams.dtype ==np.dtype('float64'):
        #     convert_back_dtype = True
        #     expparams.dtype = self.expparams_dtype

        # change_type = lambda x : x.dtype = model.expparams_dtype

        # pr0 = self.qu_likelihood(expparams["ts"], modelparams[:, np.newaxis, :])
        all_possible_exps = list(itertools.product(expparams["t"], modelparams))
        # aux = qu.parfor(self.single_qu_likelihood, all_possible_exps)
        aux = qu.parallel_map(self.single_qu_likelihood, all_possible_exps)

        splitted = np.split(np.array(aux), len(expparams))

        pr0 = np.array(splitted).reshape(-1, len(expparams))

        # if convert_back_dtype:
        #     expparams.dtype = np.dtype('float64')

        return FiniteOutcomeModel.pr0_to_likelihood_array(outcomes, pr0)

    # @classmethod
    # def qu_likelihood(cls, t, e):
    #     options = qu.Options()

    #     prob_times = []
    #     for i in t:
    #         prob_same_t = []
    #         for exps in e:
    #             H0 = 2 * np.pi * exps[0][0] * qu.sigmax()
    #             psi0 = qu.basis(2, 0)
    #             times = np.linspace(0.0, i, 1000)
    #             #Z0 is the projector to the space of |0> eigenstate
    #             Z0 = qu.ket2dm(qu.basis(2, 0))
    #             result = qu.mesolve(H0, psi0, times, [exps[0][1] * qu.sigmax()], [Z0], options=options)
    #             prob_same_t.append(result.expect[0][-1])
    #         prob_times.append(prob_same_t)
    #     return np.array(prob_times).T

    # @classmethod
    # def single_qu_likelihood(cls, par_config):
    #     final_t = par_config[0]

    #     omega = par_config[1][0]
    #     J = par_config[1][1]
    #     H0 = 2 * np.pi * omega * qu.sigmax()
    #     psi0 = qu.basis(2, 0)
    #     times = np.linspace(0.0, final_t, 3000)
    #     Z0 = qu.ket2dm(qu.basis(2, 0))
    #     result = qu.mesolve(H0, psi0, times, [J * qu.sigmax()], [Z0])
    #     return np.array(result.expect[0][-1])
    #
    #

    @classmethod
    def single_qu_likelihood(cls, par_config):
        final_t = par_config[0]

        omega = par_config[1][0]
        J = par_config[1][1]
        H0 = 2 * np.pi * omega * qu.sigmax()
        psi0 = qu.basis(2, 0)
        rho0 = qu.ket2dm(psi0)
        # times = np.linspace(0.0, final_t, 3000)
        # Z0 = qu.ket2dm(qu.basis(2, 0))
        final_dm = solve_lindblad.evolve_dm(rho0, H0, [J * qu.sigmax()], final_t, 1)
        # result = qu.mesolve(H0, psi0, times, [J * qu.sigmax()], [Z0])

        _, _, prob = qu.measurement.measurement_statistics(final_dm, qu.sigmaz())
        # return np.array(result.expect[0][-1])
        return np.array(prob[1])

    # all_possible_exps = list(itertools.product(expparams["ts"], modelparams))
    # np.split(np.array(all_possible_exps), len(expparams))

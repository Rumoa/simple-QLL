from qinfer import SMCUpdater
import numpy as np


class adaptive_SMC(SMCUpdater):
    def __init__(
        self,
        model,
        n_particles,
        prior,
        resample_a=None,
        resampler=None,
        resample_thresh=0.5,
        debug_resampling=False,
        track_resampling_divergence=False,
        zero_weight_policy="error",
        zero_weight_thresh=None,
        canonicalize=True,
        update_rate=1,
    ):
        super().__init__(
            model,
            n_particles,
            prior,
            resample_a,
            resampler,
            resample_thresh,
            debug_resampling,
            track_resampling_divergence,
            zero_weight_policy,
            zero_weight_thresh,
            canonicalize,
        )
        self.update_rate = update_rate

    def hypothetical_update(
        self, outcomes, expparams, return_likelihood=False, return_normalization=False
    ):
        """
        Produces the particle weights for the posterior of a hypothetical
        experiment.

        :param outcomes: Integer index of the outcome of the hypothetical
            experiment.
        :type outcomes: int or an ndarray of dtype int.
        :param numpy.ndarray expparams: Experiments to be used for the hypothetical
            updates.

        :type weights: ndarray, shape (n_outcomes, n_expparams, n_particles)
        :param weights: Weights assigned to each particle in the posterior
            distribution :math:`\Pr(\omega | d)`.
        """

        # It's "hypothetical", don't want to overwrite old weights yet!
        weights = self.particle_weights
        locs = self.particle_locations

        # Check if we have a single outcome or an array. If we only have one
        # outcome, wrap it in a one-index array.
        if not isinstance(outcomes, np.ndarray):
            outcomes = np.array([outcomes])

        # update the weights sans normalization
        # Rearrange so that likelihoods have shape (outcomes, experiments, models).
        # This makes the multiplication with weights (shape (models,)) make sense,
        # since NumPy broadcasting rules align on the right-most index.
        L = self.model.likelihood(outcomes, locs, expparams).transpose([0, 2, 1])
        hyp_weights = weights * np.exp(self.update_rate * np.log(L))

        # Sum up the weights to find the renormalization scale.
        norm_scale = np.sum(hyp_weights, axis=2)[..., np.newaxis]

        # As a special case, check whether any entries of the norm_scale
        # are zero. If this happens, that implies that all of the weights are
        # zero--- that is, that the hypothicized outcome was impossible.
        # Conditioned on an impossible outcome, all of the weights should be
        # zero. To allow this to happen without causing a NaN to propagate,
        # we forcibly set the norm_scale to 1, so that the weights will
        # all remain zero.
        #
        # We don't actually want to propagate this out to the caller, however,
        # and so we save the "fixed" norm_scale to a new array.
        fixed_norm_scale = norm_scale.copy()
        fixed_norm_scale[np.abs(norm_scale) < np.spacing(1)] = 1

        # normalize
        norm_weights = hyp_weights / fixed_norm_scale
        # Note that newaxis is needed to align the two matrices.
        # This introduces a length-1 axis for the particle number,
        # so that the normalization is broadcast over all particles.
        if not return_likelihood:
            if not return_normalization:
                return norm_weights
            else:
                return norm_weights, norm_scale
        else:
            if not return_normalization:
                return norm_weights, L
            else:
                return norm_weights, L, norm_scale

import qutip as qu
import numpy as np
import matplotlib.pyplot as plt
import qinfer as qi
import sys
from custom_models import  simple_precession_with_noise
import joblib

from optimization import *

from sklearn.metrics import mean_squared_error
read_cond = False
if  len(sys.argv) > 1 and (sys.argv[1] == "-s" or "--save" ):
    read_cond = True
    filename = sys.argv[2]

model = simple_precession_with_noise()
prior = qi.UniformDistribution([[0, 0],[0, 1]])
updater = qi.SMCUpdater(model, 500, prior)
# heuristic = qi.ExpSparseHeuristic(updater)
# heuristic = qi.PGH(updater, inv_field = 'x_{0}', t_field = 't')
# heuristic = qi.PGH_no_inversion(updater,  t_field="t")



# true_omegas = prior.sample()
true_omegas = np.array(np.array([[0, 0.33]]))
est_omegas = []

expdesign = qi.ExperimentDesigner(updater)
n_guesses = 1

for idx_exp in range(300):
    # experiment = heuristic()
    guesses = []
    # for i in range(n_guesses):
        # guess = np.random.exponential(scale=3*0.5, size=1)
        # guesses.append(np.random.exponential(scale=3*0.5))
        


        # guess.dtype = model.expparams_dtype
        # experiment = expdesign.design_expparams_field(guess , 't') 

    # guesses = np.array(guesses)
    # guesses.dtype = model.expparams_dtype
    # experiment = expdesign.design_expparams_field(guesses , 't') 
    # experiment.dtype = model.expparams_dtype
    guess = generate_guesses()
    # guess.dtype = model.expparams_dtype


    optimized_exp = optimize(guess, objective_fun, model, updater)

    experiment = optimized_exp['x']

    exp_before = guess
    exp_before.dtype = model.expparams_dtype

    exp_after = experiment
    exp_after.dtype = model.expparams_dtype

    print(f"Inf gain non-opt t {updater.expected_information_gain(exp_before)}.")
    print(f"Inf gain after opt {updater.expected_information_gain(exp_after)}.")


    print(f"Experimental time {experiment}")


    datum = model.simulate_experiment(true_omegas, experiment)
    updater.update(datum, experiment)
    # updater.plot_posterior_contour()

    
    # plt.plot()

    est_omegas.append(updater.est_mean())
    print(f"True parameters: {true_omegas}")
    print(f"Estimated parameters: {est_omegas[-1]}")
    print(f"Difference squared: {(est_omegas[-1] - true_omegas) ** 2}")
    # print(mean_squared_error(est_omegas[-1], true_omegas))

# print(est_omegas[-1])
# print(true_omegas)
# print(mean_squared_error(est_omegas[-1], true_omegas))
plt.semilogy((est_omegas - true_omegas) ** 2)

plt.title(f'True parameters (omega, decay_rate):{true_omegas}')
plt.xlabel('# of Measurements')
plt.ylabel('Squared Error')

plt.show()


if read_cond:
    joblib.dump([model, true_omegas, est_omegas], filename+".job")
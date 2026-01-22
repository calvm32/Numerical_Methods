from timestep_solvers import truncated_newton
import matplotlib.pyplot as plt
import numpy as np
from math import *
import random as rand

"""
Program to compare 2 different kinds of errors on the results of Newton's method,
comparing solutions only a small perturbation away
"""



# for linear regression


# -----------
# Find errors
# -----------

for N in range(N_iter):

    # slightly perturbed initial guess
    

    # lists to check tolerance and errors


    # initial zero approximate
    

    for i in range(3,18):
        tol = 10**(-i)
        N_tol_list.append(tol)

        



        # ---- consecutive diff err ----


        # machine epsilon
        eps = np.finfo(float).eps

        # protected values (same as plotting)


    # -----------
    # Plot errors
    # -----------

    # stop at machine precision


    # plot

# Fit linear regression


# Smooth x-values for line


# plot best-fit lines w small black outline

# pretend labels

plt.tight_layout()
plt.show()
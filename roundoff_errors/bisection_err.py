from timestep_solvers import bisection
import matplotlib.pyplot as plt
import numpy as np
from math import *
import random as rand

"""
Program to compare 2 different kinds of errors on the results of the bisection method,
comparing solutions only a small perturbation away
"""

N_iter = 100
opacity = 0.2
epsilon = 1e-5

f = lambda x: (cos(x) - x)

# guess interval that the root lies in
a = 0.2
b = 1

# -----------
# Setup plots
# -----------

fig, ax = plt.subplots(figsize=(10, 7))
ax.set_xlabel("tolerance"); ax.set_ylabel("error")
ax.set_title("Bisection Method Error Comparison")
ax.invert_xaxis(); ax.legend()
ax.grid(True, which="major", ls="--", alpha=0.5)

# -----------
# Find errors
# -----------

for N in range(N_iter):

    # slightly perturbed initial interval
    rand_var = epsilon*rand.uniform(-1, 1)
    a += rand_var ; b += rand_var

    # lists to check tolerance and errors
    tol_list = []
    norm_err_list = []
    diff_err_list = []

    # initial zero approximate
    prev_zero = (a + b)/2 

    for i in range(3,18):
        tol = 10**(-i)
        tol_list.append(tol)

        zero = bisection(f, a, b, tol)

        # ---- residual error ----
        norm_error = abs(f(zero))
        norm_err_list.append(norm_error)

        # ---- consecutive diff err ----
        diff_error = abs(prev_zero - zero)
        diff_err_list.append(diff_error)
        prev_zero = zero


    # -----------
    # Plot errors
    # -----------

    # stop at machine precision
    norm = np.array(norm_err_list)
    diff = np.array(diff_err_list)

    eps = np.finfo(float).eps
    norm_plot = np.maximum(norm, eps)
    diff_plot = np.maximum(diff, eps)

    # plot
    ax.loglog(tol_list, norm_plot, "-o", color="tab:blue", alpha=opacity)
    ax.loglog(tol_list, diff_plot, "-o", color="tab:orange", alpha=opacity)

# pretend labels
ax.loglog([0], [0], "-o", label=r"$|f(x_n)|$", color="tab:blue")
ax.loglog([0], [0], "-o", label=r"$|x_n - x_{n-1}|$", color="tab:orange")
ax.legend()

plt.tight_layout()
plt.show()
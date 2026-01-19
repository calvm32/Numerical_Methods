from math import *
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

def bisection(f, a, b, zero_tol=1e-8, max_iters=100):
    """
    Find a root of f(x) in the interval [a,b] using the bisection method.
    Requires [a,b] contains exactly one root that passes through the x-intersection
    """

    fa = f(a)
    fb = f(b)

    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs")

    # solve for the root
    for n in range(0, max_iters):
        mid = (a+b)/2
        fmid = f(mid)

        # quit if interval length is small enough
        if abs((b-a)/2) < zero_tol:
            return mid

        if fmid*fa < 0:
            b = mid
            fb = f(b)
        elif fmid*fb < 0:
            a = mid
            fa = f(a)
    
    return mid

# --------------
# Compare errors
# --------------

f = lambda x: (cos(x) - x)

# guess interval that the root lies in
a = 0.2
b = 1

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
fig, ax = plt.subplots(figsize=(10, 7))
ax.loglog(tol_list, norm_plot, "-o", label=r"$|f(x_n)|$", color="tab:blue")
ax.loglog(tol_list, diff_plot, "-o", label=r"$|x_n - x_{n-1}|$", color="tab:orange")

ax.set_xlabel("tolerance"); ax.set_ylabel("error")
ax.set_title("Newton's Method Error Comparison")
ax.invert_xaxis(); ax.legend()
ax.grid(True, which="major", ls="--", alpha=0.5)

plt.tight_layout()
plt.show()
from math import *
import matplotlib.pyplot as plt
import numpy as np

def newton(f, f_prime, x0, zero_tol=1e-8, max_iters=30):
    """
    Find a root of f(x) near the initial guess x0 using Newton's method
    """

    x_prev = x0

    # solve for the root
    for n in range(0, max_iters):

        x = x_prev - f(x_prev)/f_prime(x_prev)
        
        fnew = f(x)

        if abs(fnew) < zero_tol:
            return x

        x_prev = x
    
    return x

# --------------------------
# Test to find Dottie number
# --------------------------

f = lambda x: (cos(x) - x)
f_prime = lambda x: -sin(x) - 1

# initial guess
x0 = 5

# lists to check tolerance and errors
tol_list = []
norm_err_list = []
diff_err_list = []

# initial zero approximate
prev_zero = x0

for i in range(3,18):
    tol = 10**(-i)
    tol_list.append(tol)

    zero = newton(f, f_prime, x0, tol)

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
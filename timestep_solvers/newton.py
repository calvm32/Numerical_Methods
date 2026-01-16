from math import *
import matplotlib.pyplot as plt

def bisection(f, f_prime, x0, zero_tol=1e-8, max_iters=30):
    """
    Find a root of f(x) in the interval [a,b] using Newton's method.
    Requires [a,b] contains exactly one root that passes through the x-intersection
    """

    x_prev = x0

    # solve for the root
    for n in range(0, max_iters):

        x = x_prev - f(x_prev)/f_prime(x_prev)
        
        fnew = f(x)

        if abs(fnew) < zero_tol:
            return x

        x_prev = x
    
    return mid

# --------------------------
# Test to find Dottie number
# --------------------------

f = lambda x: (cos(x) - x)
f_prime = lambda x: -sin(x) - 1

# initial guess
x0 = 5

# list to check tolerance
tol_list = []
err_list = []
for i in range(3,18):
    tol = 10**(-i)
    zero = bisection(f, f_prime, x0, tol)
    error = abs(f(zero))

    tol_list.append(tol)
    err_list.append(error)

plt.figure()
plt.loglog(tol_list, err_list, "-o")
plt.gca().invert_xaxis()

plt.xlabel("tolerance")
plt.ylabel("error")
plt.tight_layout()
plt.show()
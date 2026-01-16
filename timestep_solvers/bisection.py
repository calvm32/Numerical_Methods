from math import *
import matplotlib.pyplot as plt

def bisection(f, a, b, zero_tol=1e-8, max_iters=100):
    """
    Find a root of f(x) in the interval [a,b] using Newton's method.
    Requires [a,b] contains exactly one root that passes through the x-intersection
    """

    fa = f(a)
    fb = f(b)

    # solve for the root
    for n in range(0, max_iters):
        mid = (a+b)/2
        fmid = f(mid)

        if abs(fmid) < zero_tol:
            return mid

        if fmid*fa < 0:
            b = mid
            fb = f(b)
        elif fmid*fb < 0:
            a = mid
            fa = f(a)
    
    return mid

# --------------------------
# Test to find Dottie number
# --------------------------

f = lambda x: (cos(x) - x)

# guess interval that the root lies in
a = 0.2
b = 1

# list to check tolerance
tol_list = []
err_list = []
for i in range(3,18):
    tol = 10**(-i)
    zero = bisection(f, a, b, tol)
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
from math import *
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

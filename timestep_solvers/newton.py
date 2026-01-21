from math import *
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
from math import *
import matplotlib.pyplot as plt
import math
import numpy as np

def truncate(a: float, num: int):

    truncated_a = int(a*(10**num))/(10**num)

    return truncated_a

def error_plots(f, zero, x_list, actual=float('nan')):

    abs_error = [abs(x - actual) for x in x_list]
    adj_error = [abs(x_list[i-1] - x_list[i]) for i, _ in enumerate(x_list[:-1])]
    res_error = [abs(f(x)) for x in x_list]

    # plt.figure()
    plt.semilogy(abs_error, "-o")
    plt.semilogy(adj_error, "-*")
    plt.semilogy(res_error, "-^")

    plt.xlabel("Interations")
    plt.ylabel("L2 Error")
    plt.tight_layout()
    plt.show()


# # ---------
# # Tests
# # ---------

# test_a = 0.123456789
# test_b = 10101.987898

# print(truncate(test_a, 3))
# # print(truncate(test_a, 3))
# print(truncate(test_b, 3))
# # print(truncate(test_b, 3))
# print(round(test_a, 3))
# print(round(test_b, 3))


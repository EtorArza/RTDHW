import math
from decimal import *

getcontext().prec = 50


def nCr(n, r):
    f = math.factorial
    return Decimal(f(n)) / Decimal(f(r)) / Decimal(f(n-r))


def prob_bin_n_p_k(n, p, k):
    p = Decimal(p)
    return Decimal(nCr(n, k)) * p**(k) * (1-p)**(n-k)


def upper_bound_Y_hat_leq_y_hat(target_probability, n, k):
    res = 0
    precomputed_prob_bin_n_p_k = dict()
    for v in range(0, n+1):
        for l in range(0, v - k):
            if (n, target_probability, l) not in precomputed_prob_bin_n_p_k:
                precomputed_prob_bin_n_p_k[(n, target_probability, l)] = prob_bin_n_p_k(
                    n, target_probability, l)
        mul_1 = 1 - sum([precomputed_prob_bin_n_p_k[(n, target_probability, l)]
                        for l in range(0, v - k)])
        mul_2 = prob_bin_n_p_k(n, 0.5, v)
        res += mul_1 * mul_2
    return res


def get_p_value(target_probability, n, k):
    n, k = int(n), int(k)
    assert n > 1
    assert n >= k
    assert k >= 0
    return upper_bound_Y_hat_leq_y_hat(target_probability, int(n), int(k))


def usage():
    print("This script can be used to calculate the corrected p-values. It requires three input arguments:")
    print("1) p_gamma: The desired probability of prediciting a unfairly higher runtime. It must be in the interval (0, 0.5].")
    print("2) n: The sample size.")
    print("3) k: The number of times that algorithm B did NOT outperformed algorithm A (B was executed in machine M_2 with the equivalent runtime predicted with p_gamma).")
    print("")
    print("Returns the corrected p-value of the corrected one sided sign test. The corrected test is one sided and when H_0 is rejected, accepting H_1 meanst that algorithm B is more likely to produce a lower (better) value than algorithm A.")
    print("----")
    print("Example: ")
    print("python equivalent_runtime.py 0.5 20 12")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        usage()
        exit(0)
    print('{:.8}'.format(get_p_value(float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]))))

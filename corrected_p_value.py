import math
from decimal import *

getcontext().prec = 50
max_n = 500
alpha_values = [0.05,0.01,0.001]


def inverse_bisection(f,target_f_x,a_0,b_0, target_error=0.00001):

    current_error = 10000.0

    if f(a_0) < target_f_x < f(b_0):
        a = a_0
        b = b_0
    elif f(b_0) < target_f_x < f(a_0):
        a = b_0
        b = a_0
    else:
        raise ValueError("target_f_x is not between f_a0 and f_b0:\n f_a_0, target_f_x, f_b_0: " + str(f(a_0)) + "  " + str(target_f_x) + "  " + str(f(b_0)))

    it = 0
    while current_error > target_error:
        it += 1
        
        middle = (float(a) + float(b))/2.0

        f_middle = f(middle)
        current_error = abs(f_middle - target_f_x)

        if f(middle) > target_f_x:
            b = middle
        else:
            a =  middle
    #print(it)
    return middle



def nCr(n,r):
    f = math.factorial
    # if r>n:
    #     return 0
    return Decimal(f(n)) / Decimal(f(r)) / Decimal(f(n-r))

# correction citable -> heller1986statistics
def y_r_given_z_r(n, z_r):
    return n * 0.5 +  z_r * math.sqrt(n * 0.25) - 0.5



def prob_bin_n_p_k(n,p,k):
    p = Decimal(p)
    return Decimal(nCr(n,k)) * p**(k) * (1-p)**(n-k)





def upper_bound_Y_hat_leq_y_hat(target_probability, n, k):

    res = 0

    precomputed_prob_bin_n_p_k = dict()

    for v in range(0, n+1):
        # plt.plot(list(range(0,v-k)),[prob_bin_n_p_k(n,target_probability,l) for l in range(0, v - k)])
        for l in range(0, v - k):
            if (n, target_probability, l) not in precomputed_prob_bin_n_p_k:
                precomputed_prob_bin_n_p_k[(n, target_probability, l)] = prob_bin_n_p_k(n, target_probability, l)
        mul_1 = 1 - sum([precomputed_prob_bin_n_p_k[(n, target_probability, l)] for l in range(0, v - k)])
        mul_2 = prob_bin_n_p_k(n,0.5,v)
        #print("range:",(0,n+j-k+1))
        # print((mul_1,mul_2))
        res += mul_1 * mul_2
    # plt.yscale("log")
    # plt.show()
    return res


# y = 5

# for y in range(max_n + 1):
#     print("------------")
#     #print(y_r_given_z_r(n, z_r))
#     print("n = ",max_n,", y=",y)
#     print("P(Y <= y) = ",p_Y_leq_y(max_n,y))
#     print("P(hat_Y <= y) = ",upper_bound_Y_hat_leq_y_hat(max_n,y))



# for n in range(0,10001):
#     for k in range(n // 2, -1, -1):
#         if(p_Y_leq_y(n, k) < 0.00005):
#             print(k, end=', ')
#             #print(n,"->",k)
#             break
#         if(k == 0):
#             print('INT_MAX', end=', ')
#             #print(n, "->", "INT_MAX")
#     if n% 25 == 0:
#         print('')

# exit(0)


def get_p_value(target_probability, n ,k):
    n, k = int(n), int(k)
    assert n > 1 
    assert n >= k 
    assert k >= 0
    return upper_bound_Y_hat_leq_y_hat(target_probability, int(n),int(k))
    
    





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
    print(get_p_value(float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])))

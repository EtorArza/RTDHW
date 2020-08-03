import math
from math import sqrt
from scipy.special import ndtr as ndtr

def inverse_bisection(f,target_f_x,a_0,b_0, target_error=0.00000001):

    current_error = 10000.0

    if f(a_0) < target_f_x < f(b_0):
        a = a_0
        b = b_0
    elif f(b_0) < target_f_x < f(a_0):
        a = b_0
        b = a_0
    else:
        raise ValueError("target_f_x is not between f_a0 and f_b0:\n f_a_0, target_f_x, f_b_0: " + str(f(a_0)) + "  " + str(target_f_x) + "  " + str(f(b_0)))


    while current_error > target_error:
        middle = (float(a) + float(b))/2.0

        f_middle = f(middle)
        current_error = abs(f_middle - target_f_x)

        if f(middle) > target_f_x:
            b = middle
        else:
            a =  middle
    return middle



def nCr(n,r):
    f = math.factorial
    # if r>n:
    #     return 0
    return f(n) / f(r) / f(n-r)

# correction citable -> heller1986statistics
def y_r_given_z_r(n, z_r):
    return n * 0.5 +  z_r * sqrt(n * 0.25) - 0.5


# Area to the left of this point in a normal distribution of mean 0 and variance 1
def normal_cummulative_distribution_function(x):
    return ndtr(x)


def p_Y_leq_y(n, y):

    # print(sum([0.5**n*nCr(n,i) for i in range(y+1)]))
    # target_y = y
    # f = lambda x: y_r_given_z_r(n, x)
    # target_z = inverse_bisection(f, target_y, -5, 5)
    # p = normal_cummulative_distribution_function(target_z)
    # print(p)
    # exit(1)



    if n < 25:
        return sum([prob_bin_n_p_k(n,0.5,i) for i in range(y+1)])
    else:
        target_y = y
        f = lambda x: y_r_given_z_r(n, x)
        target_z = inverse_bisection(f, target_y, -5, 5)
        p = normal_cummulative_distribution_function(target_z)
        return p

def prob_bin_n_p_k(n,p,k):
    return nCr(n,k) * p**(k) * (1-p)**(n-k)





def upper_bound_Y_hat_leq_y_hat(n, y_hat):

    res = 0

    for k in range(0, n+1):
        mul_1 = 1 - sum([prob_bin_n_p_k(n,0.01,l) for l in range(0, k - y_hat)])
        mul_2 = prob_bin_n_p_k(n,0.5,k)
        #print("range:",(0,n+j-y_hat+1))
        # print((mul_1,mul_2))
        res += mul_1 * mul_2
    return res


n = 20
# y = 5

for y in range(20):
    print("------------")
    #print(y_r_given_z_r(n, z_r))
    print("n = ",n,", y=",y)
    print("P(Y <= y) = ",p_Y_leq_y(n,y))
    print("P(hat_Y <= y) = ",upper_bound_Y_hat_leq_y_hat(n,y))
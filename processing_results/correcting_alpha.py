import math
from math import sqrt
from tqdm import tqdm as tqdm
from scipy.special import ndtr as ndtr
from decimal import *
import pandas as pd
from matplotlib import pyplot as plt

getcontext().prec = 100
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
    return n * 0.5 +  z_r * sqrt(n * 0.25) - 0.5


# Area to the left of this point in a normal distribution of mean 0 and variance 1
def normal_cummulative_distribution_function(x):
    return ndtr(x)


def p_Y_leq_y(n, y):

    # print(sum([prob_bin_n_p_k(n,0.5,i) for i in range(y+1)]))
    # target_y = y
    # f = lambda x: y_r_given_z_r(n, x)
    # target_z = inverse_bisection(f, target_y, -15, 15)
    # p = normal_cummulative_distribution_function(target_z)
    # print(p)
    # exit(1)



    if n < 200:
        return sum([prob_bin_n_p_k(n,0.5,i) for i in range(y+1)])
    else:
        target_y = y
        f = lambda x: y_r_given_z_r(n, x)
        target_z = inverse_bisection(f, target_y, -15, 15)
        p = normal_cummulative_distribution_function(target_z)
        return p

def prob_bin_n_p_k(n,p,k):
    p = Decimal(p)
    return Decimal(nCr(n,k)) * p**(k) * (1-p)**(n-k)





def upper_bound_Y_hat_leq_y_hat(n, k):

    res = 0

    for v in range(0, n+1):
        mul_1 = 1 - sum([prob_bin_n_p_k(n,0.01,l) for l in range(0, v - k)])
        mul_2 = prob_bin_n_p_k(n,0.5,v)
        #print("range:",(0,n+j-k+1))
        # print((mul_1,mul_2))
        res += mul_1 * mul_2
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






n = 30
p_values = []
corrected_p_values = []
x = list(range(n+1))
for y in tqdm(x):
    p_values.append(p_Y_leq_y(n, y))
    corrected_p_values.append(upper_bound_Y_hat_leq_y_hat(n, y))

plt.plot(x, p_values, label="$p(k)$")
plt.plot(x, corrected_p_values, label=r"$\hat{p}_c(k)$")
plt.yscale("logit")

plt.xlabel("Value of statistic, k")
plt.ylabel("p-value")

plt.legend()
plt.tight_layout()

plt.savefig("../paper/images/alpha_vs_corrected_alpha.pdf")

exit(0)

crit_values_for_each_alpha = []
for alpha in alpha_values:
    crit_values = []
    k_last = 0
    for n in tqdm(range(2, max_n + 1)):
        if k_last == 0: 
            if upper_bound_Y_hat_leq_y_hat(n,0) > alpha:
                crit_values.append("-")
                continue

        for k in range(k_last,n):
            if upper_bound_Y_hat_leq_y_hat(n,k+1) > alpha:
                crit_values.append(k)
                k_last = k
                break
    crit_values_for_each_alpha.append(crit_values)


df = pd.DataFrame(crit_values_for_each_alpha)
df = df.transpose()
df.index = [str(i) for i in range(2, max_n+1)]
df.columns =  [str(el) for el in alpha_values]

df.to_csv("/home/paran/Dropbox/BCAM/05_execution_time_as_stoping_criterion/paper/images/corrected_alpha_values.csv")
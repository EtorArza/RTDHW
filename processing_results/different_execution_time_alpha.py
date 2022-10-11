import math
from math import sqrt
from tqdm import tqdm as tqdm
from scipy.special import ndtr as ndtr
from decimal import *
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np


#execution_times = ["0.100000", "0.100001", "0.102000", "0.104000", "0.108000", "0.116000", "0.132000", "0.164000"]

execution_times = ["0.100000", "0.100001", "0.108000", "0.116000", "0.132000", "0.164000"]



def prob_bin_n_p_k(n,p,k):
    p = Decimal(p)
    return Decimal(nCr(n,k)) * p**(k) * (1-p)**(n-k)

def nCr(n,r):
    f = math.factorial
    # if r>n:
    #     return 0
    return Decimal(f(n)) / Decimal(f(r)) / Decimal(f(n-r))

def p_Y_leq_y(n, y):

    # print(sum([prob_bin_n_p_k(n,0.5,i) for i in range(y+1)]))
    # target_y = y
    # f = lambda x: y_r_given_z_r(n, x)
    # target_z = inverse_bisection(f, target_y, -15, 15)
    # p = normal_cummulative_distribution_function(target_z)
    # print(p)
    # exit(1)



    return sum([prob_bin_n_p_k(n,0.5,i) for i in range(y+1)])

def process_chunck(data_chunck):
    res = []
    for time in execution_times[1:]:
        res.append(is_h0_rejected(data_chunck, time))
    return res

def is_h0_rejected(data_chunck, execution_time):
    A_samples = get_samples_of_certain_time(data_chunck, "0.100000")
    B_samples = get_samples_of_certain_time(data_chunck, execution_time)
    
    

    n = 0
    y = 0
    for a_i, b_i in zip(A_samples, B_samples):
        if -a_i == -b_i:
            continue
        elif -a_i < -b_i:
            y += 1
        n += 1
    if p_Y_leq_y(n,y) < 0.05:
        return True
    else:
        return False

def get_samples_of_certain_time(data_chunck, execution_time):
    result = []
    for line in data_chunck:
        if "1|"+execution_time in line:
            result.append(float(line.split("|")[-1]))
    return result


list_of_lines = []

with open("linear_regression_calibration/result_diff_execution_time.txt") as f:
    for line in f:
        list_of_lines.append(line.strip())

dataset_list = []
number_of_rejects = np.array([0 for el in execution_times[1:]])
number_of_chuncks = 0
for line in tqdm(list_of_lines):
    if "---" in line:
        number_of_chuncks+= 1
        number_of_rejects+= np.array(process_chunck(dataset_list)).astype(int)
        dataset_list = []
    else:
        dataset_list.append(line)

percentage_of_rejects = number_of_rejects / number_of_chuncks

x = (np.array([float(el) for el in execution_times[1:]]) - 0.1) /0.1 * 100
x_labels = [str(round(el))+"%" for el in x]
y = percentage_of_rejects

print(np.stack((x,y)))


plt.plot(x,y,marker="o")
plt.xticks(x, labels=x_labels)
plt.xlabel("Percentage of extra execution time")
plt.ylabel("Probability of type I error")
plt.ylim(bottom=0)

plt.tight_layout()
plt.savefig("figures/extra_execution_time_typeI_error.pdf")
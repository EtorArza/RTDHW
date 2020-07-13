import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm as tqdm

columns=["cpuname","problemtype","problempath","methodname","operatorname","nevals","maxtime","maxevals","time","evals","fitness", "taskname"]


lines = []
with open("../result.txt", "r") as f:
    for line in f:
        line = line.strip()
        line = line.split("|")
        line = line[0:5] + [int(line[5])] + [float(line[6])] + [int(line[7])] + [float(line[8])] + [int(line[9])] + [float(line[10])] + [line[2].split("/")[-1] + "_" + line[4]]
        lines.append(line)

df = pd.DataFrame(lines, columns=columns)





# We say that the proportion between two tasks is constant, lets say it is k. 
# The confidence interva is defined as k * (1 - c interval perc) < x < k *(1 + c_interval perc)
# Since k is estimated as an average obtained in all cpus, this function returns the number of times that the actual constant k is within these two values.
def what_percentage_corresponds_to_the_c_inteval(df, c_interval_perc):
    tasks = df["taskname"].unique()

    cases_within_interval = 0
    cases_outside_interval = 0

    for i, task_i in enumerate(tasks):
        for j, task_j in enumerate(tasks):
            if i > j:
                continue
            sub_df_0 = df[df["taskname"]==task_i].reset_index(drop=True)
            sub_df_1 = df[df["taskname"]==task_j].reset_index(drop=True)
            res_df = sub_df_0["time"]/sub_df_1["time"]
            upper_bound = res_df.mean() * (1 + c_interval_perc)
            lower_bound = res_df.mean() * (1 - c_interval_perc)

            for k in res_df:
                if lower_bound < k < upper_bound:
                    cases_within_interval+=1
                else:
                    cases_outside_interval+=1
    return float(cases_within_interval) / (cases_outside_interval + cases_within_interval)



y_vec = [0.497, 0.41, 0.36, 0.29, 0.17]
x_vec = [1.0 - what_percentage_corresponds_to_the_c_inteval(df, item) for item in tqdm(y_vec)]
target_x = [0.001, 0.01, 0.05, 0.1, 0.2]

print("These two vectors below should be equal. Change y_vec so that they are.")
print(x_vec)
print(target_x)
print("##################################################")

for i in range(len(y_vec)):
    x = i
    y = y_vec[i]

    plt.plot([x, x, x], [1.0-y ,1.0, 1.0+y], color="black")
    plt.scatter([x, x], [1.0-y, 1.0+y], marker="_", color="black")

plt.yticks([0.0, 1.0, 2.0], labels=["0","k","2k"])
plt.xticks([float(i) for i in range(len(y_vec))], labels=["$"+str(item)+"$" for item in target_x])
#plt.xscale("log")
plt.xlabel("Error of the confidence interval")
plt.ylabel("Size of the confidence interval")
plt.show()

plt.close()


######################### REGRESION ########


cpunames = df["cpuname"].unique()
print(cpunames)





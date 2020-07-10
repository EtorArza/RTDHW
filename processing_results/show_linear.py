import pandas as pd
from matplotlib import pyplot as plt

columns=["cpuname","problemtype","problempath","methodname","operatorname","nevals","maxtime","maxevals","time","evals","fitness"]


lines = []
with open("../result.txt", "r") as f:
    for line in f:
        line = line.strip()
        line = line.split("|")
        line = line[0:5] + [int(line[5])] + [float(line[6])] + [int(line[7])] + [float(line[8])] + [int(line[9])] + [float(line[10])]
        lines.append(line)

df = pd.DataFrame(lines, columns=columns)

cpunames = df["cpuname"].unique()

sub_df_0 = df[(df["operatorname"]=="exchange") & (df["problempath"]=="instances/qap/tai100a.qap")].reset_index(drop=True)
sub_df_1 = df[(df["operatorname"]=="swap") & (df["problempath"]=="instances/lop/N-t65d11xx_150.lop")].reset_index(drop=True)
sub_df_2 = df[(df["operatorname"]=="insert") & (df["problempath"]=="instances/pfsp/tai200_20_0.pfsp")].reset_index(drop=True)
sub_df_3 = df[(df["operatorname"]=="random_search") & (df["problempath"]=="instances/tsp/kroB200.tsp")].reset_index(drop=True)


list_of_subdf = [sub_df_0, sub_df_1, sub_df_2, sub_df_3]

res_df = sub_df_0["time"]/sub_df_1["time"]

x = 0
for i in range(len(list_of_subdf)):
    for j in range(len(list_of_subdf)):
        if i >= j:
            continue
        res_df = list_of_subdf[i]["time"]/list_of_subdf[j]["time"]
        x_pos = [x for _ in range(len(list_of_subdf[0]))]
        plt.scatter(x_pos, res_df, marker=".")
        for y,cpuname in zip(res_df, cpunames):
            plt.text(x_pos[0], y, cpuname + "  " + str((i,j)))
        x += 1


        

plt.show()

print(list_of_subdf)
print("------")
print(sub_df_0["time"],sub_df_1["time"])
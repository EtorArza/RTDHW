import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm as tqdm
from statistics import mean,median
from matplotlib.ticker import PercentFormatter
from matplotlib import rc




columns=["cpuname","problemtype","problempath","methodname","operatorname","nevals","maxtime","maxevals","time","evals","fitness", "taskname","normalizedtime"]

lines = []
with open("../result.txt", "r") as f:
    for line in f:
        line = line.strip()
        line = line.split("|")
        line = line[0:5] + [int(line[5])] + [float(line[6])] + [int(line[7])] + [float(line[8])] + [int(line[9])] + [float(line[10])] + [line[2].split("/")[-1] + "_" + line[4]] + [0.0]

        lines.append(line)

df = pd.DataFrame(lines, columns=columns)

for taskname in df["taskname"].unique():
    norm_times = df[df["taskname"] == taskname]["time"] / df[df["taskname"] == taskname]["time"].mean()
    df["normalizedtime"][norm_times.index] = norm_times

######################### REGRESION PLOT #########################



cpunames = df["cpuname"].unique()
cpu_passmark_single_thread_scores = {
    'i5_470U_1_33gh':411,
    'i7_2760QM_2_4gh':1550,
    'intel_celeron_n4100_1_1gh':1032,
    'ryzen7_1800X':2176,
    'i7_7500U_2_7gh':2025,
    'amd_fx_6300_hexacore':1484
}








################# PLOT Confidence intervals of constant k #################

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


def get_all_k_constants(df):
    tasks = df["taskname"].unique()

    cases_within_interval = 0
    cases_outside_interval = 0

    k_values = []

    for i, task_i in enumerate(tasks):
        for j, task_j in enumerate(tasks):
            if i <= j:
                continue
            sub_df_0 = df[df["taskname"]==task_i].reset_index(drop=True)
            sub_df_1 = df[df["taskname"]==task_j].reset_index(drop=True)
            res_df = sub_df_0["time"]/sub_df_1["time"]


            for k in res_df / res_df.mean():
                k_values.append(k)
    return np.array(k_values)



print("$$$$ correlation $$$$$$$$$$$$")





tasks = df["taskname"].unique()
sorted_df = df.sort_values(by='cpuname', inplace=False, axis = 0)
machines_in_cols_and_tasks_in_rows = []

for task in tasks:
    machines_in_cols_and_tasks_in_rows.append(df[df["taskname"] == task]["time"].to_list())

machines_in_cols_and_tasks_in_rows = pd.DataFrame(machines_in_cols_and_tasks_in_rows)

print(machines_in_cols_and_tasks_in_rows.corr(method="pearson"))
print(machines_in_cols_and_tasks_in_rows.corr(method="spearman"))
print(machines_in_cols_and_tasks_in_rows.corr(method="kendall"))

exit(1)

###########################################33


print("$$$$ HISTOGRAM k $$$$$$$$$$$$")


k_values = get_all_k_constants(df)
N = 10
delta = 0.05
bins=sorted([(1-delta)**n for n in range(1,N)])+[1]+sorted([(1+delta)**n for n in range(1,N)])
k_values_clipped = np.clip(k_values, bins[0], bins[-1])
plt.hist(k_values_clipped, bins=bins,edgecolor='black', linewidth=1.2,  weights=np.ones(len(k_values_clipped)) / len(k_values_clipped))
plt.xscale("log")
n_ticks = 6
xticks = [(bins[1]+bins[0])/2]+[bins[int((i * len(bins))//n_ticks)] for i in range(1,n_ticks)]+[(bins[-1]+bins[-2])/2]

labels = ["< "+str(round(bins[0],2))]+["{:.2f}".format(item) for item in xticks[1:-1]]+ ["> "+str(round(bins[-1],2))]


for percentage in [0.2, 0.5]:
    print(len(k_values[(k_values < 1.0+percentage) & (k_values > 1-percentage)])/len(k_values),"percent of the cases are within", percentage , "of the average" )

plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.xticks(xticks, labels=labels)
plt.xlabel(r"percentual deviation: $r_j(A,B) \ / \ \bar{r}(A,B$)", fontsize=14)
plt.ylabel("percentage of cases")
plt.tight_layout()
plt.savefig("../../paper/images/histogram_k.pdf")
plt.close()


print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")




y_vec = [0.354, 0.287, 0.173, 0.073]
x_vec = [1.0 - what_percentage_corresponds_to_the_c_inteval(df, item) for item in tqdm(y_vec)]
target_x = [0.05, 0.1, 0.2, 0.5]

print("##################################################")
print("These two vectors below should be equal. Change y_vec so that they are.")
print(x_vec)
print(target_x)
print("##################################################")

for i in range(len(y_vec)):
    x = i
    y = y_vec[i]

    plt.plot([x, x, x], [1.0-y ,1.0, 1.0+y], color="black")
    plt.scatter([x, x], [1.0-y, 1.0+y], marker="_", color="black")

plt.yticks([0.0, 1.0, 2.0], labels=["0","r","2r"])
plt.xticks([float(i) for i in range(len(y_vec))], labels=["$"+str(item)+"$" for item in target_x])
#plt.xscale("log")
plt.xlabel("Error of the confidence interval")
plt.ylabel("Size of the confidence interval")
plt.tight_layout()
plt.savefig("../../paper/images/ci_of_constant_ratio_k.pdf")

plt.close()




#########################################################################









# compute x_vec -> passmark_single_thread_score and y_vec -> average_normalized_runtime
def get_regression_x_y_from_df(training_df):
    x_vec=[]
    y_vec=[]
    y_vec_norm=[]

    for cpuname in training_df["cpuname"].unique():
        passmark_single_thread_score = cpu_passmark_single_thread_scores[cpuname]
        total_time_cpu = training_df[training_df["cpuname"]==cpuname]["time"].mean()
        total_time_cpu_norm = training_df[training_df["cpuname"]==cpuname]["normalizedtime"].mean()

        x_vec.append(passmark_single_thread_score)
        y_vec.append(total_time_cpu)
        y_vec_norm.append(total_time_cpu_norm)
    return x_vec, y_vec_norm, y_vec

# same function as get_regression_x_y_from_df() but returns a value for each row in the df, 
# while get_regression_x_y_from_df() averages times by cpu
def get_test_x_y(test_df):
    res = map(get_regression_x_y_from_df, [test_df.iloc[i:] for i in range(df.shape[0])])
    x_vec_test, _, y_vec_test = [],None,[]

    for i in range(df.shape[0]):
        row = test_df.iloc[i:]
        res = get_regression_x_y_from_df(row)
        x_vec_test += res[0]
        y_vec_test += res[2]

    return x_vec_test, None, y_vec_test



# returns y_test = g(x) = a * x_test + b  |  where a and b are fitted with training_df
def fit_and_predict(training_df, x_test):

    x_vec, y_vec_norm, _ = get_regression_x_y_from_df(training_df)
    coef = np.polyfit(x_vec,y_vec_norm,1)
    poly1d_fn = np.poly1d(coef)
    return poly1d_fn(x_test)

x_vec, y_vec_norm, _ = get_regression_x_y_from_df(df)

res = fit_and_predict(df,[0,1])

b = res[0]
a = res[1] - b

plt.plot(x_vec,y_vec_norm, 'rx')
plt.plot([min(x_vec)*0.6, max(x_vec)*1.1], fit_and_predict(df, [min(x_vec)*0.6, max(x_vec)*1.1]), '--k', label="$t(x) = "+"{:.6f}".format(a)+"x + "+"{:.4f}".format(b)+"$")


plt.legend()
plt.xlabel("Passmark single thread score, $x$")
plt.ylabel("Time, $t(A_0)$")

plt.tight_layout()
plt.savefig("../../paper/images/passmark_base_algorithm_regression.pdf")
plt.close()
##### Leave same category out cross validation #####




# Compute the percentage of cases in which y_test[i] < y_test_pred[i] is true
def what_percentage_of_predicted_in_regression_is_within_c_interval(df_train, x_test, y_test, CORRECTION_COEFFICIENT):



    g_x_predictions = fit_and_predict(df_train, x_test)

    cases_within_interval = 0
    cases_outside_interval = 0
    #print(["g_x1", "g_x2", "t_prd", "t_actual"])

    perc_time_diff = []
    for test_index_1 in range(len(x_test)):
        for test_index_2 in range(len(x_test)):
            if test_index_1 == test_index_2:
                continue
            g_x1 = g_x_predictions[test_index_1]
            g_x2 = g_x_predictions[test_index_2]
            t_pred = g_x2 / g_x1 * y_test[test_index_1] * CORRECTION_COEFFICIENT
            t_actual = y_test[test_index_2]

            #print(["{:.2f}".format(a_float) for a_float in [g_x1, g_x2, t_pred, t_actual]])
            perc_time_diff.append(t_pred / t_actual)
            if t_pred < t_actual:
                cases_within_interval+=1
            else:
                cases_outside_interval+=1

    return float(cases_within_interval) / (cases_within_interval + cases_outside_interval), mean(perc_time_diff)




def df_wraper_what_percentage_of_predicted_in_regression_is_within_c_interval(df_train, df_test, CORRECTION_COEFFICIENT):
    x_test, _ , y_test = get_test_x_y(df_test)
    return what_percentage_of_predicted_in_regression_is_within_c_interval(df_train, x_test, y_test, CORRECTION_COEFFICIENT)


CORRECTION_COEFFICIENTS = [1.0 - 0.025 / (2 ** i) for i in range(6,1,-1)]+[1.0 - 0.025*i for i in range(1,28)]

def test_a_CORRECTION_COEFFICIENT(CORRECTION_COEFFICIENT_in):
    CORRECTION_COEFFICIENT = CORRECTION_COEFFICIENT_in
    pred_lower_actual_cases_perc_list = []
    perc_time_predicted_with_respect_to_actual_list = []
    for problem in ("qap", "lop", "pfsp", "tsp"):
        for i,hw_1 in enumerate(cpunames):
            for j,hw_2 in enumerate(cpunames):
                if i>=j:
                    continue
                sub_df_train = df[ (df["problemtype"]!=problem) & (df["cpuname"]!=hw_1) & (df["cpuname"]!=hw_2) ]
                sub_df_test = df[ (df["problemtype"]==problem) & ((df["cpuname"]==hw_1) | (df["cpuname"]== hw_2)) ]
                pred_lower_cases_perc, perc_time_predicted_with_respect_to_actual = df_wraper_what_percentage_of_predicted_in_regression_is_within_c_interval(sub_df_train, sub_df_test, CORRECTION_COEFFICIENT)
                pred_lower_actual_cases_perc_list.append(pred_lower_cases_perc)
                perc_time_predicted_with_respect_to_actual_list.append(perc_time_predicted_with_respect_to_actual)

    perc_cases_pred_lower = mean(pred_lower_actual_cases_perc_list)
    pred_time_in_average_this_percent_lower_than_actual = mean(perc_time_predicted_with_respect_to_actual_list)
    return CORRECTION_COEFFICIENT, perc_cases_pred_lower, pred_time_in_average_this_percent_lower_than_actual
    # print("Due to the selected correction value of ", CORRECTION_COEFFICIENT, ":")
    # print("The percentage of cases in which the predicted time was lower than the actual time is: ", mean(pred_lower_actual_cases_perc_list))
    # print("The predicted time was in average ", mean(perc_time_predicted_with_respect_to_actual_list), " percent of the actual time.")


perc_cases_pred_lower_list = []
pred_time_in_average_this_percent_lower_than_actual_list = []
for CC in tqdm(CORRECTION_COEFFICIENTS):
    CORRECTION_COEFFICIENT, perc_cases_pred_lower, pred_time_in_average_this_percent_lower_than_actual = test_a_CORRECTION_COEFFICIENT(CC)
    perc_cases_pred_lower_list.append(perc_cases_pred_lower)
    pred_time_in_average_this_percent_lower_than_actual_list.append(pred_time_in_average_this_percent_lower_than_actual)

perc_cases_pred_higher_list = 1 - np.array(perc_cases_pred_lower_list)

print("The correction coefficients:",CORRECTION_COEFFICIENTS)
print("Percentage of cases in which pred was higher:", perc_cases_pred_higher_list)
print("On average, the predicted time was this much percent lower",pred_time_in_average_this_percent_lower_than_actual_list)
plt.plot(CORRECTION_COEFFICIENTS, perc_cases_pred_higher_list, label=r"$P(\hat{t}_2(B) > t_2(B))$",linestyle="-")
plt.plot(CORRECTION_COEFFICIENTS, pred_time_in_average_this_percent_lower_than_actual_list, label=r"$\mathbb{E}(\dfrac{\hat{t}_2(B)}{t_2(B)})$",linestyle="--")

plt.legend()
plt.ylabel("")
plt.xlabel("correction parameter  $\gamma$")
plt.ylim((0,1.05))
plt.tight_layout()
plt.savefig("../../paper/images/correction_coefficient_tradeoff.pdf")
plt.close()





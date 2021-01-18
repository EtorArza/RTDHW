import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm as tqdm
from statistics import mean, median
from matplotlib.ticker import PercentFormatter
from matplotlib import rc
import seaborn as sns


columns = ["cpuname", "problemtype", "problempath", "methodname", "operatorname", "nevals",
           "maxtime", "maxevals", "time", "evals", "fitness", "taskname"]

lines = []
with open("result.txt", "r") as f:
    for line in f:
        line = line.strip()
        line = line.split("|")
        line = line[0:5] + [int(line[5])] + [float(line[6])] + [int(line[7])] + [float(line[8])] + [
            int(line[9])] + [float(line[10])] + [line[2].split("/")[-1] + "_" + line[4]]

        lines.append(line)

df = pd.DataFrame(lines, columns=columns)


######################### REGRESION PLOT #########################


cpunames = df["cpuname"].unique()
cpu_passmark_single_thread_scores = {
    'i5_470U_1_33gh': 411,
    'i7_2760QM_2_4gh': 1563,
    'intel_celeron_n4100_1_1gh': 1032,
    'ryzen7_1800X': 2180,
    'i7_7500U_2_7gh': 2025,
    'amd_fx_6300_hexacore': 1484,
    'AMD_A9_9420_RADEON_R5':1311,
    'i7_6700HQ_CPU_2_60GHz_bisk':1918,

}


def inverse_bisection(f, target_f_x, a_0, b_0, target_error=0.00000001):

    current_error = 10000.0

    if f(a_0) < target_f_x < f(b_0):
        a = a_0
        b = b_0
    elif f(b_0) < target_f_x < f(a_0):
        a = b_0
        b = a_0
    else:
        raise ValueError("target_f_x is not between f_a0 and f_b0:\n f_a_0, target_f_x, f_b_0: " +
                         str(f(a_0)) + "  " + str(target_f_x) + "  " + str(f(b_0)))

    it = 0
    while current_error > target_error:
        it += 1

        middle = (float(a) + float(b))/2.0

        f_middle = f(middle)
        current_error = abs(f_middle - target_f_x)

        if f(middle) > target_f_x:
            b = middle
        else:
            a = middle
    return middle


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
            sub_df_0 = df[df["taskname"] == task_i].reset_index(drop=True)
            sub_df_1 = df[df["taskname"] == task_j].reset_index(drop=True)
            res_df = sub_df_0["time"]/sub_df_1["time"]
            upper_bound = res_df.mean() * (1 + c_interval_perc)
            lower_bound = res_df.mean() * (1 - c_interval_perc)

            for k in res_df:
                if lower_bound < k < upper_bound:
                    cases_within_interval += 1
                else:
                    cases_outside_interval += 1
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
            sub_df_0 = df[df["taskname"] == task_i].reset_index(drop=True)
            sub_df_1 = df[df["taskname"] == task_j].reset_index(drop=True)
            res_df = sub_df_0["time"]/sub_df_1["time"]

            for k in res_df / res_df.mean():
                k_values.append(k)
    return np.array(k_values)


print("$$$$ correlation $$$$$$$$$$$$")


tasks = df["taskname"].unique()
sorted_df = df.sort_values(by='cpuname', inplace=False, axis=0)
machines_in_cols_and_tasks_in_rows = []

for task in tasks:
    machines_in_cols_and_tasks_in_rows.append(
        df[df["taskname"] == task]["time"].to_list())

machines_in_cols_and_tasks_in_rows = pd.DataFrame(
    machines_in_cols_and_tasks_in_rows)

mask_ut = np.triu(np.ones(machines_in_cols_and_tasks_in_rows.corr(
    method="pearson").shape)).astype(np.bool)


for metric, title in zip(["pearson", "spearman", "kendall"], ["Pearson's correlation coefficient", "Spearman's rank correlation coefficient", r"Kendal-$\tau$ rank correlation coeficient"]):
    corr = machines_in_cols_and_tasks_in_rows.corr(method=metric)

    sorted_df_lower_triangle = corr.where(
        np.triu(np.ones(corr.shape) - np.eye(corr.shape[0])).astype(np.bool))

    print("Average", title, np.nanmean(sorted_df_lower_triangle.values))

    axes = sns.heatmap(corr, mask=mask_ut, cmap="viridis", vmin=0, vmax=1)
    plt.xlabel("machine index")
    plt.ylabel("machine index")
    plt.title(title)
    plt.tight_layout()
    axes.figure.savefig("../paper/images/" + metric + ".pdf")
    plt.close()

# hmap.figure.savefig("Correlation_Heatmap_Lower_Triangle_with_Seaborn_using_mask.png",
#                    format='png',
#                    dpi=150)


print("explaining graph on prediction of equivalent time")



t_1_B, t_2_B = 1, 2.75
t_1_A, t_2_A = 0.5, 1.5
t_1_C, t_2_C = t_1_A, t_2_A
t_1_D, t_2_D = 1.5, 3.5



x_lims = (0, 2.3)
y_lims = (0, 4.5)


def fx_actual(x): return (t_2_D - t_2_A)/(t_1_D - t_1_A)*x + t_2_A - ((t_2_D - t_2_A)/(t_1_D - t_1_A)*t_1_A)


def fx_predicted(x): return (0 - t_2_A)/(0 - t_1_A)*x + t_2_A - ((0 - t_2_A)/(0 - t_1_A)*t_1_A)


delta = 0.05


def draw_grid_on_point(x, y):
    plt.hlines(y, 0, x, linestyles="--", linewidths=0.5)
    plt.vlines(x, 0, y, linestyles="--", linewidths=0.5)


# setup graph and background
plt.xlim(x_lims)
plt.ylim(y_lims)
# plt.hlines(0,*y_lims)
# plt.vlines(0,*x_lims)


# draw interpolation
plt.plot([*x_lims], list(map(fx_actual, [*x_lims])), linestyle="--", linewidth=1.0, c="tab:orange", label="Prediction with two references")
plt.plot([*x_lims], list(map(fx_predicted, [*x_lims])), linestyle="-.",
         linewidth=1.0, c="tab:blue", label="Prediction with one reference")


# draw points and add text
plt.scatter(t_1_A, t_2_A, marker="x", c="tab:red", label="Runtime of tasks")
plt.text(t_1_A+delta, t_2_A+delta, r"  $\rho'$")
plt.scatter(t_1_D, t_2_D, marker="x", c="tab:red")
plt.text(t_1_D+delta, t_2_D+delta, r"  $\rho''$")
plt.scatter(t_1_B, t_2_B, marker="x", c="tab:red")
plt.text(t_1_B+delta, t_2_B+delta, r"  $\rho$")
plt.scatter(t_1_B, fx_predicted(t_1_B), marker="s", c="tab:blue",
            linewidths=2.0, label=r"Predicted runtime of the runtime of $\rho$ in machine\n $M_2$ with one reference")
plt.scatter(t_1_B, fx_actual(t_1_B), marker=".", c="tab:orange",
            linewidths=3.75, label=r"Predicted runtime of the runtime of $\rho$ in machine\n $M_2$ with two references")

# plt.scatter(t_1_B, fx_predicted(t_1_B), marker=".", c="tab:blue", linewidths=0.30)
# plt.scatter(t_1_B, fx_actual(t_1_B), marker=".", c="tab:orange", linewidths=0.30)
# plt.text(0.2, fx_predicted(t_1_B)+delta, "predicted runtime of task $B$ in machine $P_2$")

# add grid to points
draw_grid_on_point(t_1_A, t_2_A)
#draw_grid_on_point(t_1_B, t_2_B)
draw_grid_on_point(t_1_D, t_2_D)
draw_grid_on_point(t_1_B, fx_predicted(t_1_B))
draw_grid_on_point(t_1_B, fx_actual(t_1_B))


# labels on tics
plt.xticks([0, t_1_A, t_1_B, t_1_D], labels=["0", r" $t(M_1,\rho')$", r"$t(M_1,\rho)$", r"$t(M_1,\rho'')$"])
plt.yticks([t_2_A, fx_actual(t_1_B), t_2_B, fx_predicted(t_1_B), t_2_D], labels=[
           r"  $t(M_2,\rho')$",  r"  ", r"  $t(M_2,\rho)$", r"  ", r"  $t(M_2,\rho'')$", ])


# labels on axes
plt.xlabel("Runtimes in machine $M_1$")
plt.ylabel("Runtimes in machine $M_2$")

plt.legend()
plt.legend(loc=4, prop={'size': 9})
plt.tight_layout()
plt.savefig("../paper/images/explaination_prediction.pdf")
plt.close()


print("$$$$ HISTOGRAM k $$$$$$$$$$$$")


k_values = get_all_k_constants(df)
N = 10
delta = 0.05
bins = sorted([(1-delta)**n for n in range(1, N)])+[1] + \
    sorted([(1+delta)**n for n in range(1, N)])
k_values_clipped = np.clip(k_values, bins[0], bins[-1])
plt.hist(k_values_clipped, bins=bins, edgecolor='black', linewidth=1.2,
         weights=np.ones(len(k_values_clipped)) / len(k_values_clipped))
plt.xscale("log")
n_ticks = 6
xticks = [(bins[1]+bins[0])/2]+[bins[int((i * len(bins))//n_ticks)]
                                for i in range(1, n_ticks)]+[(bins[-1]+bins[-2])/2]

labels = ["< "+str(round(bins[0], 2))]+["{:.2f}".format(item)
                                        for item in xticks[1:-1]] + ["> "+str(round(bins[-1], 2))]


for percentage in [0.2, 0.5]:
    print(len(k_values[(k_values < 1.0+percentage) & (k_values > 1-percentage)]) /
          len(k_values), "percent of the cases are within", percentage, "of the average")

plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.xticks(xticks, labels=labels)
plt.xlabel(r"percentual deviation: $r_j(A,B) \ / \ \bar{r}(A,B$)", fontsize=14)
plt.ylabel("percentage of cases")
plt.tight_layout()
plt.savefig("../paper/images/histogram_k_LEGACY.pdf")
plt.close()


print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")


y_vec = [0.354, 0.287, 0.173, 0.073]
x_vec = [
    1.0 - what_percentage_corresponds_to_the_c_inteval(df, item) for item in tqdm(y_vec)]
target_x = [0.05, 0.1, 0.2, 0.5]

print("##################################################")
print("These two vectors below should be equal. Change y_vec so that they are. (LEGACY)")
print(x_vec)
print(target_x)
print("##################################################")

for i in range(len(y_vec)):
    x = i
    y = y_vec[i]

    plt.plot([x, x, x], [1.0-y, 1.0, 1.0+y], color="black")
    plt.scatter([x, x], [1.0-y, 1.0+y], marker="_", color="black")

plt.yticks([0.0, 1.0, 2.0], labels=["0", "r", "2r"])
plt.xticks([float(i) for i in range(len(y_vec))], labels=[
           "$"+str(item)+"$" for item in target_x])
# plt.xscale("log")
plt.xlabel("Error of the confidence interval")
plt.ylabel("Size of the confidence interval")
plt.tight_layout()
plt.savefig("../paper/images/ci_of_constant_ratio_k.pdf")

plt.close()


#########################################################################




def get_test_x_y(test_df):


    if test_df['cpuname'].unique().shape[0] != 2:
        print('Error, more than two machines in get test x_y(). n of machines was ', test_df['cpuname'].unique().shape[0])



    if test_df['cpuname'].unique().shape:
        pass

    x_vec_test = [cpu_passmark_single_thread_scores[item] for item in test_df['cpuname']]
    y_vec_test = test_df['time'].to_list()
    y_vec_ref = [float(test_df[(test_df['taskname'] == taskname_item) & (test_df['cpuname'] != cpuname_item)]['time']) for taskname_item, cpuname_item in zip(test_df['taskname'], test_df['cpuname'])]

    #print(y_vec_ref)

    return x_vec_test, y_vec_ref, y_vec_test


# returns y_test = g(x) = a * x_test + b  |  where a and b are fitted with training_df
def fit_and_predict(training_df, x_test):

    x_vec = [cpu_passmark_single_thread_scores[item] for item in training_df['cpuname'].unique()]
    y_vec_norm = []

    for cpuname in training_df['cpuname'].unique():
        y_vec_norm.append(training_df[training_df['cpuname'] == cpuname]['time'].sum())
    
    
    coef = np.polyfit(x_vec, y_vec_norm, 1)
    poly1d_fn = np.poly1d(coef)
    return poly1d_fn(x_test)

x_vec = [cpu_passmark_single_thread_scores[item] for item in df['cpuname'].unique()]
y_vec_norm = []

for cpuname in df['cpuname'].unique():
    y_vec_norm.append(df[df['cpuname'] == cpuname]['time'].sum())


res = fit_and_predict(df, [0, 1])

b = res[0]
a = res[1] - b

plt.plot(x_vec, y_vec_norm, 'rx')
plt.plot([min(x_vec)*0.6, max(x_vec)*1.1], fit_and_predict(df, [min(x_vec)*0.6, max(x_vec)
                                                                * 1.1]), '--k', label=r"$t(s_j) = "+"{:.6f}".format(a)+r"s_j + "+"{:.4f}".format(b)+"$")


plt.legend()
plt.xlabel(r"Machine score, $s_j$")
plt.ylabel(r"Runtime of $\rho'$, $t(M_j,\rho')$")

plt.tight_layout()
plt.savefig("../paper/images/passmark_base_algorithm_regression.pdf")
plt.close()
##### Leave same category out cross validation #####


# Compute the percentage of cases in which y_test[i] < y_test_pred[i] is true
def what_percentage_of_predicted_in_regression_is_within_c_interval(df_train, x_test, y_test, y_P_1_s_prima_ref, CORRECTION_COEFFICIENT):

    g_x_predictions = fit_and_predict(df_train, x_test)

    cases_within_interval = 0
    cases_outside_interval = 0
    #print(["g_x1", "g_x2", "t_prd", "t_actual"])

    # print(x_test)
    # print(y_test)

    perc_time_diff = []
    for index in range(len(x_test)):
        g_x2 = g_x_predictions[index]
        g_x1 = [item for item in g_x_predictions if item != g_x2][0] # select the machine score that was not assigned to g_x2. The prediction is made in machine P_2, so the index corresponds to machine P_2.
        t_pred_P_2_s = g_x2 / g_x1 * y_P_1_s_prima_ref[index] * CORRECTION_COEFFICIENT
        t_actual_P_2_s = y_test[index]
        # print('scoreP2,    ScoreP1,    refP1,   pred,   actual, CORRECTION_COEF')
        # print(["{:.2f}".format(a_float) for a_float in [g_x2, g_x1, y_P_1_s_prima_ref[index], t_pred_P_2_s, t_actual_P_2_s, CORRECTION_COEFFICIENT]])
        perc_time_diff.append(t_pred_P_2_s / t_actual_P_2_s)
        if t_pred_P_2_s < t_actual_P_2_s:
            cases_within_interval += 1
        else:
            cases_outside_interval += 1
    return float(cases_within_interval) / (cases_within_interval + cases_outside_interval), mean(perc_time_diff)


def df_wraper_what_percentage_of_predicted_in_regression_is_within_c_interval(df_train, df_test, CORRECTION_COEFFICIENT):
    x_test, y_P_1_s_prima_ref, y_test = get_test_x_y(df_test)
    return what_percentage_of_predicted_in_regression_is_within_c_interval(df_train, x_test, y_test, y_P_1_s_prima_ref, CORRECTION_COEFFICIENT)


CORRECTION_COEFFICIENTS = [
    1.0 - 0.025 / (2 ** i) for i in range(6, 1, -1)]+[1.0 - 0.025*i for i in range(1, 28)]


def test_a_CORRECTION_COEFFICIENT(CORRECTION_COEFFICIENT_in):
    CORRECTION_COEFFICIENT = CORRECTION_COEFFICIENT_in
    pred_lower_actual_cases_perc_list = []
    perc_time_predicted_with_respect_to_actual_list = []
    for problem in ("qap", "lop", "pfsp", "tsp"):
        for i, hw_1 in enumerate(cpunames):
            for j, hw_2 in enumerate(cpunames):
                if i >= j:
                    continue
                sub_df_train = df[(df["problemtype"] != problem) & (df["cpuname"] != hw_1) & (df["cpuname"] != hw_2)]
                sub_df_test = df[(df["problemtype"] == problem) & ((df["cpuname"] == hw_1) | (df["cpuname"] == hw_2))]

                sub_df_train.reset_index(drop=True, inplace=True)
                sub_df_test.reset_index(drop=True, inplace=True)
                #print('sub_df_train',sub_df_train)
                #print('sub_df_test', sub_df_test)
                pred_lower_cases_perc, perc_time_predicted_with_respect_to_actual = df_wraper_what_percentage_of_predicted_in_regression_is_within_c_interval(
                    sub_df_train, sub_df_test, CORRECTION_COEFFICIENT)
                pred_lower_actual_cases_perc_list.append(pred_lower_cases_perc)
                perc_time_predicted_with_respect_to_actual_list.append(
                    perc_time_predicted_with_respect_to_actual)

    perc_cases_pred_lower = mean(pred_lower_actual_cases_perc_list)
    pred_time_in_average_this_percent_lower_than_actual = mean(
        perc_time_predicted_with_respect_to_actual_list)
    return CORRECTION_COEFFICIENT, perc_cases_pred_lower, pred_time_in_average_this_percent_lower_than_actual
    # print("Due to the selected correction value of ", CORRECTION_COEFFICIENT, ":")
    # print("The percentage of cases in which the predicted time was lower than the actual time is: ", mean(pred_lower_actual_cases_perc_list))
    # print("The predicted time was in average ", mean(perc_time_predicted_with_respect_to_actual_list), " percent of the actual time.")


perc_cases_pred_lower_list = []
pred_time_in_average_this_percent_lower_than_actual_list = []
for CC in tqdm(CORRECTION_COEFFICIENTS):
    CORRECTION_COEFFICIENT, perc_cases_pred_lower, pred_time_in_average_this_percent_lower_than_actual = test_a_CORRECTION_COEFFICIENT(
        CC)
    perc_cases_pred_lower_list.append(perc_cases_pred_lower)
    pred_time_in_average_this_percent_lower_than_actual_list.append(
        pred_time_in_average_this_percent_lower_than_actual)

perc_cases_pred_higher_list = 1 - np.array(perc_cases_pred_lower_list)

print("The correction coefficients:", CORRECTION_COEFFICIENTS)
print("Percentage of cases in which pred was higher:", perc_cases_pred_higher_list)
print("On average, the predicted time was this much percent lower",
      pred_time_in_average_this_percent_lower_than_actual_list)
plt.plot(CORRECTION_COEFFICIENTS, perc_cases_pred_higher_list,
         label=r"$P(\hat{t}_2 > t_2)$", linestyle="-")
# plt.plot(CORRECTION_COEFFICIENTS, pred_time_in_average_this_percent_lower_than_actual_list,
#          label=r"$\mathbb{E}[\dfrac{\hat{t}_2}{t_2}]$", linestyle="--")

# plt.plot(CORRECTION_COEFFICIENTS, CORRECTION_COEFFICIENTS,
#          label=r"$\gamma$", linestyle="-.")

# plt.legend()
plt.ylabel(r"$P(\hat{t}_2 > t_2)$")
plt.xlabel("correction parameter  $\gamma = \mathbb{E}[\dfrac{\hat{t}_2}{t_2}]$")
plt.ylim((0, 1.05))
plt.tight_layout()
plt.savefig("../paper/images/correction_coefficient_tradeoff.pdf")
plt.close()

print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

print("Add to article preface:")

max_prob_longer = 0.01

gamma = 0
expected_runtime = 0

for i in range(len(perc_cases_pred_higher_list)):
    if perc_cases_pred_higher_list[i] < max_prob_longer:
        gamma = CORRECTION_COEFFICIENTS[i]
        expected_runtime = pred_time_in_average_this_percent_lower_than_actual_list[i]
        break
        



print(r'\newcommand\nmachines{',len(cpunames),'}',sep='')
print(r'\newcommand\ntasks{',len(tasks),'}',sep='')
print(r'\newcommand\aparamregression{',"{:.5f}".format(a),'}',sep='')
print(r'\newcommand\bparamregression{',round(b),'}',sep='')
print(r'\newcommand\bdivaparamregression{',"{:.2f}".format(b/a),'}',sep='')
print(r'\newcommand\minusbdivaparamregression{',"{:.2f}".format(-b/a),'}',sep='')
print(r'\newcommand\chosengammavalue{',gamma,'}',sep='')
print(r'\newcommand\expectedruntimeratiowithchosengamma{',"{:.3f}".format(expected_runtime),'}',sep='')
print(r'\newcommand\factormultiplytimesexampleI{',"{:.3f}".format(gamma*(-b/a - cpu_passmark_single_thread_scores['intel_celeron_n4100_1_1gh']) / (-b/a - 1230)),'}',sep='')
print(r'\newcommand\factormultiplytimesexampleII{',"{:.3f}".format(gamma*(-b/a - cpu_passmark_single_thread_scores['intel_celeron_n4100_1_1gh']) / (-b/a - cpu_passmark_single_thread_scores['ryzen7_1800X'])),'}',sep='')



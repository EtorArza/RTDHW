import pandas as pd
import sys
import numpy as np
sys.path.append(".")
import equivalent_runtime
from sklearn.neighbors import KernelDensity
from matplotlib import pyplot as plt


# Read original data

columns = ["cpuname", "problemtype", "problempath", "methodname", "operatorname", "nevals",
           "maxtime", "maxevals", "time", "evals", "fitness", "taskname", "cpuscore"]


cpu_passmark_single_thread_scores = {
    'i5_470U_1_33gh': 539,
    'i7_2760QM_2_4gh': 1559,
    'intel_celeron_n4100_1_1gh': 1012,
    'ryzen7_1800X': 2185,
    'i7_7500U_2_7gh': 1955,
    'amd_fx_6300_hexacore': 1486,
    'AMD_A9_9420_RADEON_R5':1344,
    'i7_6700HQ_CPU_2_60GHz_bisk':1921,
}



lines = []
with open("linear_regression_calibration/result.txt", "r") as f:
    for line in f:
        line = line.strip()
        line = line.split("|")
        line = line[0:5] + [int(line[5])] + [float(line[6])] + [int(line[7])] + [float(line[8])] + [
            int(line[9])] + [float(line[10])] + [line[2].split("/")[-1] + "_" + line[4]] + [cpu_passmark_single_thread_scores[line[0]]]

        lines.append(line)

df_fit = pd.DataFrame(lines, columns=columns)




print(df_fit["cpuname"].unique())

list_of_rows = []
for cpuname in df_fit["cpuname"].unique():
    cpuscore=df_fit.query(f"cpuname == '{cpuname}'")["cpuscore"].unique()[0]
    sorted_cpu_df = df_fit.query(f"cpuname == '{cpuname}'").sort_values(by="taskname")
    tasks = list(sorted_cpu_df["taskname"])
    list_of_rows.append([cpuname, cpuscore] + list(sorted_cpu_df["time"]))

    
df_fit = pd.DataFrame(list_of_rows, columns=["cpuname", "passmark"] + tasks)


# # The scope of these changes made to
# # pandas settings are local to with statement.
# with pd.option_context('display.max_rows', None,
#                        'display.max_columns', 70,
#                        'display.precision', 3,
#                        ):
#     print(df_fit)


# Read verification data

df_test = pd.read_csv("verification_experiments/result.csv", header=None, names=["cpuname", "passmark", "time1", "time2", "time3", "time4"], )

for kernel_size in [0.05, 0.0001]:
    for function, function_label in zip([lambda x: x, lambda x: np.abs(x)], ["", "abs"]):
        for plot_index, figures_name in zip(range(3),("fitdata_eqivruntime_vs_runtime", "verification_eqivruntime_vs_runtime", "eqivruntime_fit_vs_verification")):


            fig, ax = plt.subplots()
            fig_cum, ax_cum = plt.subplots()
            for df, label, data_name in zip([df_fit, df_test], ["Fit data", "Verification data"], ["fitdata", "equivruntime"]):
                proportions_equivalent_runtime = []
                proportions_same_runtime = []
                for i in range(len(df)):
                    for j in range(len(df)):
                        if i < j:
                            continue
                        s1 = df.iloc[i]["passmark"]
                        s2 = df.iloc[j]["passmark"]

                        print(f"$M_{i+1} \\rightarrow M_{j+1}$")
                        print(df.iloc[i]["cpuname"], "->", df.iloc[j]["cpuname"])

                        tasknames = list(df.columns)
                        tasknames.remove('cpuname')
                        tasknames.remove('passmark')

                        for task in tasknames:    
                            t1 = df.iloc[i][task]
                            t2 = df.iloc[j][task]

                            estimated_t2 = equivalent_runtime.get_equivalent_runtime_from_probability(0.5, s1, s2, t1)

                            proportions_equivalent_runtime.append(estimated_t2 / t2)
                            proportions_same_runtime.append(t1 / t2)

                nslices = 100000
                x_plot = np.linspace(-1,1,nslices)

                def get_y_plot_from_proportions(proportions, x_plot):
                    log_proportions =  function(np.log10(np.array(proportions)))
                    kde = KernelDensity(kernel='tophat', bandwidth=kernel_size).fit(log_proportions.reshape(-1, 1))
                    y_plot =  np.exp(kde.score_samples(x_plot.reshape(-1, 1)))
                    return y_plot


                y_plot_equivalent_runtime = get_y_plot_from_proportions(proportions_equivalent_runtime, x_plot)
                y_plot_same_runtime = get_y_plot_from_proportions(proportions_same_runtime, x_plot)




                # Uses trapezoid rule to get empirical distribution from tophat kde. Requires small tophat and a huge nslices.
                def get_empirical(x_plot, y_plot):
                    return np.concatenate(([0],(np.cumsum(y_plot[1:-1]) + (y_plot[2:] + y_plot[0])/2) * (x_plot[2] -  x_plot[1]), [1]))


                if data_name in figures_name:
                    ax.plot(x_plot, y_plot_equivalent_runtime, label=label+" equivalent runtime")                
                    ax_cum.plot(x_plot, get_empirical(x_plot, y_plot_equivalent_runtime), label=label+" equivalent runtime")

                    if "runtime" in figures_name:
                        ax.plot(x_plot, y_plot_same_runtime, label=label+" same runtime")
                        ax_cum.plot(x_plot, get_empirical(x_plot, y_plot_same_runtime), label=label+" same runtime")






            ax.legend()
            fig.savefig(f"figures/verif_{figures_name}_kde{function_label}_kernelsize_{kernel_size}.pdf")

            ax_cum.set_xlim((-0.1, 1))
            ax_cum.legend()
            if function_label == "abs":
                fig_cum.savefig(f"figures/verif_{figures_name}_cumulative_kernelsize_{kernel_size}_.pdf")



    
    


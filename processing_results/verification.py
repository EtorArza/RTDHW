import pandas as pd
import sys
import numpy as np
sys.path.append(".")
import equivalent_runtime
from sklearn.neighbors import KernelDensity
from matplotlib import pyplot as plt






# plot goodies from # https://stackoverflow.com/questions/7358118/matplotlib-black-white-colormap-with-dashes-dots-etc

def setAxLinesBW(ax):
    """
    Take each Line2D in the axes, ax, and convert the line style to be 
    suitable for black and white viewing.
    """
    MARKERSIZE = 3

    COLORMAP = {
        '#1f77b4': {'marker': None, 'dash': [5,3]},
        '#ff7f0e': {'marker': None, 'dash': (None,None)},
        '#2ca02c': {'marker': None, 'dash': [5,3,1,3]},
        '#d62728': {'marker': None, 'dash': [1,3]},
        '#9467bd': {'marker': None, 'dash': [5,2,5,2,5,10]},
        '#8c564b': {'marker': None, 'dash': [5,3,1,2,1,10]},
        '#e377c2': {'marker': 'o', 'dash': (None,None)} #[1,2,1,10]}
        }


    lines_to_adjust = ax.get_lines()
    try:
        lines_to_adjust += ax.get_legend().get_lines()
    except AttributeError:
        pass

    print("Adjusting colors: ")
    for line in lines_to_adjust:
        origColor = line.get_color()
        print(origColor)
        # line.set_color('black') # Uncomment for b&w figure
        line.set_dashes(COLORMAP[origColor]['dash'])
        line.set_marker(COLORMAP[origColor]['marker'])
        line.set_markersize(MARKERSIZE)

def setFigLinesBW(fig):
    """
    Take each axes in the figure, and for each line in the axes, make the
    line viewable in black and white.
    """
    for ax in fig.get_axes():
        setAxLinesBW(ax)










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

for kernel_size in [0.1, 0.0001]:
    for function, function_label in zip([lambda x: x, lambda x: np.abs(x)], ["", "abs"]):
        for plot_index, figures_name in zip(range(3),("fitdata_equiv_vs_runtime", "verification_equiv_vs_runtime", "equiv_fitdata_vs_verification")):


            fig, ax = plt.subplots()
            fig_cum, ax_cum = plt.subplots()
            for df, label, data_name in zip([df_fit, df_test], ["Fit data", "Verification data"], ["fitdata", "verification"]):
                proportions_equivalent_runtime = []
                proportions_same_runtime = []
                for i in range(len(df)):
                    for j in range(len(df)):

                        if df.iloc[i]["cpuname"] == df.iloc[j]["cpuname"]:
                            continue

                        s1 = df.iloc[i]["passmark"]
                        s2 = df.iloc[j]["passmark"]

                        # print(f"$M_{i+1} \\rightarrow M_{j+1}$")
                        # print(df.iloc[i]["cpuname"], "->", df.iloc[j]["cpuname"])

                        tasknames = list(df.columns)
                        tasknames.remove('cpuname')
                        tasknames.remove('passmark')

                        for task in tasknames:    
                            t1 = df.iloc[i][task]
                            t2 = df.iloc[j][task]

                            estimated_t2 = equivalent_runtime.get_equivalent_runtime_from_probability(0.5, s1, s2, t1)

                            proportions_equivalent_runtime.append(estimated_t2 / t2)
                            proportions_same_runtime.append(t1 / t2)

                x_min, x_max = (-2,2)
                nslices = 100000
                x_plot = np.linspace(x_min,x_max,nslices)

                def get_y_plot_from_proportions(proportions, x_plot):
                    log_proportions =  function(np.log2(np.array(proportions)))
                    kde = KernelDensity(kernel='tophat', bandwidth=kernel_size).fit(log_proportions.reshape(-1, 1))
                    y_plot =  np.exp(kde.score_samples(x_plot.reshape(-1, 1)))
                    return y_plot


                y_plot_equivalent_runtime = get_y_plot_from_proportions(proportions_equivalent_runtime, x_plot)
                y_plot_same_runtime = get_y_plot_from_proportions(proportions_same_runtime, x_plot)




                # Uses trapezoid rule to get empirical distribution from tophat kde. Requires small tophat and a huge nslices.
                def get_empirical(x_plot, y_plot):
                    return np.concatenate(([0],(np.cumsum(y_plot[1:-1]) + (y_plot[2:] + y_plot[0])/2) * (x_plot[2] -  x_plot[1]), [1]))


                if data_name in figures_name:

                    if "runtime" in figures_name:
                        ax.plot(x_plot, y_plot_same_runtime, label="No runtime adjustment")
                        ax_cum.plot(x_plot, get_empirical(x_plot, y_plot_same_runtime), label="No runtime adjustment")


                        ax.plot(x_plot, y_plot_equivalent_runtime, label="Equivalent runtime")
                        ax_cum.plot(x_plot, get_empirical(x_plot, y_plot_equivalent_runtime), label="Equivalent runtime")

                    else:
                        ax.plot(x_plot, y_plot_equivalent_runtime, label=label)
                        ax_cum.plot(x_plot, get_empirical(x_plot, y_plot_equivalent_runtime), label=label)





            fig_path_prefix = f"figures/verif_{plot_index}_{figures_name}_"

            setAxLinesBW(ax)
            setAxLinesBW(ax_cum)

            ax.legend()
            if "abs" not in function_label:
                fig.savefig(fig_path_prefix + f"kde{function_label}_kernelsize_{kernel_size}.pdf")

            ax_cum.legend()
            ax_cum.set_xlim((-0.1, x_max))
            ax_cum.set_xlabel(r"Log$_2$ deviation ratio")
            ax_cum.set_ylabel("Cumulative probability")
            if function_label == "abs" and kernel_size < 0.01:
                fig_cum.savefig(fig_path_prefix + f"cumulative_kernelsize_{kernel_size}_.pdf")

            plt.close(fig)
            plt.close(fig_cum)



    
    


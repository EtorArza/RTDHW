import subprocess
import pandas as pd



res_file = "case_study/comparison_with_hamming_eda/res_EDA.csv"
instances_dir = "case_study/comparison_with_hamming_eda/instances/"
csv_path = "case_study/comparison_with_hamming_eda/stopping_criterion.csv"


def write_conf_file(PROBLEM_PATH, MAX_SOLVER_TIME):
    with open("tmp.ini", "w") as f:
        print(
"""
[Global]
SOLVER_NAME = iterated_local_search
OPERATOR = exchange
USE_TABU = false
MAX_SOLVER_TIME = {MAX_SOLVER_TIME}
MAX_SOLVER_EVALS = -1
N_EVALS = 20
N_OF_THREADS = 1

PROBLEM_TYPE = qap
PROBLEM_PATH = {PROBLEM_PATH}

RESET_METHOD = reinitialize
STEPS_TO_MOVE_TOWARDS_RANDOM = 0
SEED = 2""".format(**{"PROBLEM_PATH": PROBLEM_PATH, "MAX_SOLVER_TIME": str(MAX_SOLVER_TIME)}), file=f)


def get_result(PROBLEM_PATH, MAX_SOLVER_TIME):
    subprocess.call("touch comp_result.txt", shell=True)
    subprocess.call("rm comp_result.txt", shell=True)
    write_conf_file(PROBLEM_PATH, MAX_SOLVER_TIME)
    res = subprocess.call("./main.out tmp.ini comp_result.txt dummy_machine", shell=True)
    with open("comp_result.txt","r") as f:
        res = f.readlines()[0].strip().split("|")[-1]
        return res



df = pd.read_csv(csv_path)


for row_index in range(df.shape[0]):
    instance_name, runtime_hat_t_2 = df.iloc[row_index, :]
    print(instance_name)
    with open(res_file, "a") as res_file_obj:    	
        print(instance_name, runtime_hat_t_2, end=",", sep=",", file=res_file_obj)
        print(get_result(instances_dir+instance_name+".qap", str(runtime_hat_t_2)), file=res_file_obj)

#instance_path_list = [instances_dir+el for el in str(subprocess.check_output("ls {instances_dir}".format(instances_dir=instances_dir), shell=True)).strip("'b").split("\\n") if el != ""]



import equivalent_runtime

with open("verification_experiments/result.txt", "r") as f:
    all_lines = f.readlines()
    name_list = []
    passmark_list = []
    task1_time_list = []
    task2_time_list = []
    for i in range(len(all_lines) / 5):
        

#!/bin/bash

cd linear_regression_calibration
make

progressbar()
{
    total_steps=$2
    bar="#########################"
    barlength=${#bar}
    n=$(($1*barlength/$total_steps))
    printf "\r[%-${barlength}s (%d%%)] " "${bar:0:n}" "$(( $1*100/$total_steps ))" 

    if [[ "$1" == "$2" ]]; then
        echo "done."
    fi
}

echo -n "initialization...  "
./main.out initial_execution.ini initialization.txt "dummy_cpu_name"
echo initialized.

i=0

END=1000

for ((REPS=1;REPS<=END;REPS++)); do
for PROBLEM_TYPE in "qap" "tsp" "pfsp" "lop"; do
for PROBLEM_PATH in "instances/${PROBLEM_TYPE}/"* ; do
for OPERATOR in "random_search"; do
for MAX_SOLVER_TIME in "0.100000" "0.100001" "0.102000" "0.104000" "0.108000" "0.116000" "0.132000" "0.164000"; do



i=$(($i+1))

progressbar $i $((16*8*$END))


if [[ "$OPERATOR" == "random_search" ]]; then
    SOLVER_NAME="random_search"
else    
    SOLVER_NAME="iterated_local_search"
fi

SEED=$i

cat > tmp.ini <<EOF
[Global]
SOLVER_NAME = $SOLVER_NAME
OPERATOR = $OPERATOR
USE_TABU = false
MAX_SOLVER_TIME = ${MAX_SOLVER_TIME}
MAX_SOLVER_EVALS = 2000000000
TABU_LENGTH = 40
N_EVALS = 1
N_OF_THREADS = 1

PROBLEM_TYPE = ${PROBLEM_TYPE}
PROBLEM_PATH = ${PROBLEM_PATH}

RESET_METHOD = reinitialize 
STEPS_TO_MOVE_TOWARDS_RANDOM = 0
SEED = ${SEED}
EOF

./main.out tmp.ini result_diff_execution_time.txt "dummy_cpu_name"

done
done
done
done

echo "---" >> result_diff_execution_time.txt
done









sudo apt install -y g++
sudo apt install -y make

make clean
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
./main.out initial_execution.ini initialization.txt
echo initialized.

i=0
for PROBLEM_TYPE in "qap" "tsp" "pfsp" "lop"; do
for PROBLEM_PATH in "instances/${PROBLEM_TYPE}/"* ; do
for OPERATOR in "swap" "exchange" "insert" "random_search"; do

i=$(($i+1))

progressbar $i 160


if [[ "$OPERATOR" == "random_search" ]]; then
    SOLVER_NAME="random_search"
else    
    SOLVER_NAME="iterated_local_search"
fi



cat > tmp.ini <<EOF
[Global]
SOLVER_NAME = $SOLVER_NAME
OPERATOR = $OPERATOR
USE_TABU = true
MAX_SOLVER_TIME = 99999999.9
MAX_SOLVER_EVALS = 1000000
TABU_LENGTH = 40
N_EVALS = 1
N_OF_THREADS = 1

PROBLEM_TYPE = ${PROBLEM_TYPE}
PROBLEM_PATH = ${PROBLEM_PATH}

RESET_METHOD = reinitialize 
STEPS_TO_MOVE_TOWARDS_RANDOM = 0
SEED = 2
EOF

./main.out tmp.ini result.txt

done
done
done











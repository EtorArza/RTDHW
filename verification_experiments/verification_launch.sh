#!/bin/bash


g++ main_verification_1_primes.cpp -o verification1.out
g++ main_verification_2_magic_squares.cpp -o verification2.out
g++ main_verification_3_knapsack.cpp -o verification3.out

chmod u+x verification1.out
chmod u+x verification2.out
chmod u+x verification3.out

echo "the processor name is: "
cat /proc/cpuinfo | grep "model name" | head -1
cat /proc/cpuinfo | grep "model name" | head -1 >> result.csv


START="$(date +%s%N)"
./verification1.out
DURATION=$[ $(date +%s%N) - ${START} ]
echo ${DURATION}
echo -n ",${DURATION}," >> result.csv
echo "-----"

START="$(date +%s%N)"
./verification2.out
DURATION=$[ $(date +%s%N) - ${START} ]
echo ${DURATION}
echo "${DURATION}," >> result.csv
echo "-----"



START="$(date +%s%N)"
./verification3.out
DURATION=$[ $(date +%s%N) - ${START} ]
echo ${DURATION}
echo "${DURATION}," >> result.csv
echo "-----"







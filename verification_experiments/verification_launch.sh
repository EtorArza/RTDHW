#!/bin/bash


g++ main_verification_1_primes.cpp -o verification1.out
g++ main_verification_2_magic_squares.cpp -o verification2.out


echo "the processor name is: "
cat /proc/cpuinfo | grep "model name" | head -1
cat /proc/cpuinfo | grep "model name" | head -1 >> result.txt


START="$(date +%s%N)"
./verification1.out
DURATION=$[ $(date +%s%N) - ${START} ]
echo ${DURATION}
echo ${DURATION} >> result.txt
echo "-----"

START="$(date +%s%N)"
./verification2.out
DURATION=$[ $(date +%s%N) - ${START} ]
echo ${DURATION}
echo ${DURATION} >> result.txt
echo "-----"






echo "-----" >> result.txt

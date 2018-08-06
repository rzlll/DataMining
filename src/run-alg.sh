#!/usr/bin/env bash

if [ $# -ne 2 ]; then
    echo "usage: run-alg.sh alg data"
    exit 1
fi

alg=$1
data=$2

OUTPUT_DIR="../${alg}res/$data/"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
    echo "create $OUTPUT_DIR"
fi

for k in $(seq 0 10)
do
    for N in $(seq 0 10)
    do
        for ind in $(seq 0 19)
        do
            echo ${alg}run.py $data $k $N $ind
            python ${alg}run.py $data $k $N $ind &
        done
        wait;
    done
done

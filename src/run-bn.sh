#!/usr/bin/env bash

data=$1

OUTPUT_DIR="../bnres/$data/"
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
            echo $data $k $N $ind
            python bnrun.py $data $k $N $ind &
        done
        wait;
    done
done

#!/usr/bin/env bash

if [ $# -le 1 ]; then
    echo "usage: run-alg.sh alg [data+]"
    exit 1
fi

alg=$1

function run() {
    data=$1
    echo "run $alg $data"
    OUTPUT_DIR="../${alg}res/$data/"
    if [ ! -d $OUTPUT_DIR ]; then
        mkdir -p $OUTPUT_DIR
        echo "create $OUTPUT_DIR"
    fi

    for k in $(seq 0 10)
    do
        for N in $(seq 0 10)
        do
            for ind in $(seq 11 30)
            do
                echo ${alg}run.py $data $k $N $ind
                python ${alg}run.py $data $k $N $ind &
            done
            wait;
            for ind in $(seq 31 49)
            do
                echo ${alg}run.py $data $k $N $ind
                python ${alg}run.py $data $k $N $ind &
            done
            wait;
        done
    done
}

shift
for var in "$@"
do
    run $var
done

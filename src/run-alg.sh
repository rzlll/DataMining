#!/usr/bin/env bash

if [ $# -le 2 ]; then
    echo "usage: run-alg.sh algorithm ncores [data+]"
    exit 1
fi

alg=$1
ncore=$2

begin=0
ending=49

function run_single() {
    data=$1
    k=$2
    N=$3
    ind=$4

    if [ $alg = 'rev2' ]
    then
        echo ${alg}run.py $data 1 1 1 1 1 1 0 50 $k $N $ind
        python ${alg}run.py $data 1 1 1 1 1 1 0 50 $k $N $ind
    else
        echo ${alg}run.py $data $k $N $ind
        python ${alg}run.py $data $k $N $ind
    fi
}

function run_parallel() {
    data=$1
    OUTPUT_DIR="../${alg}res/$data/"
    if [ ! -d $OUTPUT_DIR ]; then
        mkdir -p $OUTPUT_DIR
        echo "create $OUTPUT_DIR"
    fi

    for k in $(seq 0 10)
    do
        for N in $(seq 0 10)
        do
            for inds in $(seq $begin $ncore $ending)
            do
                inde=$(($inds + $ncore - 1))
                inde=$(($inde<$ending?$inde:$ending))
                for ind in $(seq $inds $inde)
                do
                    run_single $data $k $N $ind &
                done
                wait;
            done
        done
    done
}

shift; shift;

for var in "$@"
do
    echo "run $alg $ncore $data"
    run_parallel $var
    echo "finish $alg $ncore $data"
done


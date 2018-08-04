#!/usr/bin/env bash

NET=$1
k=$2
n=$3
it=$4
inds=$5
inde=$6

for c1 in $(seq 1 2)
do
    for c2 in $(seq 1 2)
    do
        for c3 in $(seq 1 2)
        do
            for c4 in $(seq 1 2)
            do
                for c5 in $(seq 1 2)
                do
                    for c6 in $(seq 1 2)
                    do
                        for c7 in $(seq 0 0)
                        do
                            for ind in $(seq $inds $inde)
                            do
                                python rev2run.py $NET $c1 $c2 $c3 $c4 $c5 $c6 $c7 $it $k $n $ind &
                                echo $NET $c1 $c2 $c3 $c4 $c5 $c6 $c7 $it $k $n $inds $inde
                            done
                            wait;
                        done
                        wait;
                    done
                    wait;
                done
                wait;
            done
            #wait;
        done
        #wait;
    done
done

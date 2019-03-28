#!/usr/bin/env bash

NET=$1

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
                            python algs/rev2run.py $NET $c1 $c2 $c3 $c4 $c5 $c6 $c7 $it &
                            echo algs/rev2run.py $NET $c1 $c2 $c3 $c4 $c5 $c6 $c7 $it
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

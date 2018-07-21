#!/usr/bin/env bash
# Generate fake data

for data in alpha amazon epinions otc; do
    for k in $(seq 0 20); do
        for n in $(seq 0 10); do
            echo "$data, $k, $n"
            python ./fake.py $data $k $n &
        done
        wait
    done
done

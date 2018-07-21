#!/usr/bin/env bash
# Run rev2 on all the data (with/without fake data)

for data in alpha amazon epinions otc; do
    for k in $(seq 0 20); do
        for n in $(seq 0 10); do
            echo "$data, $k, $n"
            OUTPUT_DIR="../res/fake-$data-$k-$n/"
            if [ ! -d $OUTPUT_DIR ]; then
                mkdir $OUTPUT_DIR
            fi
            bash run-rev2-all-params.sh $data $k $n
        done
    done
done

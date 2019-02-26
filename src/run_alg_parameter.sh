#!/usr/bin/env bash

if [ $# -le 2 ]; then
    echo "usage: run-alg_parameter.sh algorithm ncores [data+]"
    exit 1
fi

alg=$1
ncores=$2

begin=0
ending=19    #upscale to ending=49

function run_single() {
	data=$1
	k=$2
	N=$3
	ind=$4

	if [ $alg = 'fraudar' ] || [ $alg = 'freagle' ] || [ $alg = 'rsd' ]
	then
		echo algs/${alg}.py $data $k $N $ind $n_pattern
        python algs/${alg}.py $data $k $N $ind $n_pattern
    fi
}

function run_parallel() {
	data=$1
    OUTPUT_DIR="../res/${alg}/$data/"

    if [ ! -d $OUTPUT_DIR]; then
    	mkdir -p $OUTPUT_DIR
    	echo "create $OUTPUT_DIR"
   	fi

   	


}



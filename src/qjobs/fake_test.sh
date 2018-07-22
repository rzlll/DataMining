#!/usr/bin/env bash
# Run rev2 on all the data (with/without fake data)

# The name of the job, can be anything, simply used when displaying the list of running jobs
#$ -N fake-test
# Combining output/error messages into one file
#$ -j y
# Set memory request:
#$ -l vf=45G
# Set walltime request:
#$ -l h_rt=10:59:59
# One needs to tell the queue system to use the current directory as the working directory
# Or else the script may fail as it will execute in your top level home directory /home/username
#$ -cwd
# then you tell it retain all environment variables (as the default is to scrub your environment)
#$ -V
# Now comes the command to be executed
source $HOME/venv/bin/activate

data=$1

# for data in alpha amazon epinions otc; do
for k in $(seq 0 10); do
    for n in $(seq 0 10); do
        echo "$data, $k, $n"
        OUTPUT_DIR="../rev2res/fake-$data-$k-$n/"
        if [ ! -d $OUTPUT_DIR ]; then
            mkdir $OUTPUT_DIR
        fi
        bash run-rev2-all-params.sh $data $k $n 100
    done
done
wait
# done
exit 0

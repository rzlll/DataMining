#!/usr/bin/env python3

sample='''
#!/usr/bin/env bash
# Run rev2 on all the data (with/without fake data)

# The name of the job, can be anything, simply used when displaying the list of running jobs
#$ -N rev2-$data-$k-$n
# Combining output/error messages into one file
#$ -j y
# Set memory request:
#$ -l vf=2G
#$ -pe smp 10
# Set walltime request:
#$ -l h_rt=1:59:59
# One needs to tell the queue system to use the current directory as the working directory
# Or else the script may fail as it will execute in your top level home directory /home/username
#$ -cwd
# then you tell it retain all environment variables (as the default is to scrub your environment)
#$ -V
# Now comes the command to be executed
source $HOME/venv/bin/activate

# for data in alpha amazon epinions otc; do

echo "$data, $k, $n"
OUTPUT_DIR="../rev2res/fake-$data-$k-$n/"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir $OUTPUT_DIR
fi
bash run-rev2-all-params.sh $data $k $n 20

wait
# done
exit 0
'''

for d in ['alpla', 'amazon', 'epinions', 'otc']:
    for k in range(11):
        for n in range(11):
            tmp = sample.replace('$k', str(k)).replace('$n', str(n)).replace('$data', d)
            with open('rev2test-%s-%d-%d.qjob' %(d, k, n), 'w') as f:
                f.write(tmp)

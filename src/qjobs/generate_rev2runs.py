#!/usr/bin/env python3

import sys, os
import argparse

sample='''
#!/usr/bin/env bash
# Run rev2 on all the data (with/without fake data)

# The name of the job, can be anything, simply used when displaying the list of running jobs
#$ -N $qjob_name
# Combining output/error messages into one file
#$ -j y

# Set memory request:
#$ -l vf=1G

# ironfs access
##$ -l ironfs

# number of processes (cores)
#$ -pe smp 8

# Set walltime request:
#$ -l h_rt=3:59:59

# One needs to tell the queue system to use the current directory as the working directory
# Or else the script may fail as it will execute in your top level home directory /home/username
#$ -cwd
# then you tell it retain all environment variables (as the default is to scrub your environment)
#$ -V
# Now comes the command to be executed
source $HOME/venv/bin/activate
cd $HOME/research/fake-review/src

echo "$data, $k, $n"
OUTPUT_DIR="../rev2res/$data/"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi
bash run-rev2-all-params.sh $data $k $n 50 $inds $inde

wait
# done
exit 0
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Exploits pages')
    parser.add_argument('-d', '--data', action='store', choices=['alpha', 'amazon', 'epinions', 'otc'], required=True, help='target dataset')
    parser.add_argument('-i', '--index', action='store', nargs=2, required=True, help='start and end index of target product')
    parsed = parser.parse_args(sys.argv[1:])
    
    print(parsed)
    
    step = 10
    d = parsed.data
    i_s = int(parsed.index[0])
    i_e = int(parsed.index[1])
    
    for k in range(11):
        for n in range(11):
            for inds in range(i_s, i_e, step):
                qjob_name = 'rev2run-%s-%d-%d-%d.qjob' %(d, k, n, inds)
                tmp = sample.replace('$k', str(k)).replace('$n', str(n)).replace('$data', d).replace('$inds', str(inds)).replace('$inde', str(inds + step - 1)).replace('$qjob_name', qjob_name)
                with open(qjob_name, 'w') as f:
                    f.write(tmp)

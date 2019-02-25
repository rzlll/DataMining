#!/usr/bin/env python3

import sys, os
import argparse
import subprocess

anthill_template = '''
#!/usr/bin/env bash

# The name of the job, can be anything, simply used when displaying the list of running jobs
#$ -N rev2-$data-$k-$n-$ind
# Combining output/error messages into one file
#$ -j y

# Set memory request:
#$ -l vf=$vf

# ironfs access
###$ -l ironfs

# number of processes (cores)
### -pe smp 8

# Set walltime request:
#$ -l h_rt=10:59:59

# One needs to tell the queue system to use the current directory as the working directory
# Or else the script may fail as it will execute in your top level home directory /home/username
###$ -cwd
#$ -wd /home/ifsdata/scratch/rliu/qlog/
# then you tell it retain all environment variables (as the default is to scrub your environment)
####$ -V
# Now comes the command to be executed

# activate virtual env for python3

source $HOME/venv/bin/activate
cd $HOME/research/fake-review/src

OUTPUT_DIR="../res/rev2/$data/"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

python algs/rev2run.py $data 1 1 1 1 1 1 0 10 $k $n $ind
python algs/rev2run.py $data 1 2 1 1 1 1 0 10 $k $n $ind
python algs/rev2run.py $data 1 1 2 1 1 1 0 10 $k $n $ind
python algs/rev2run.py $data 1 1 1 2 1 1 0 10 $k $n $ind
python algs/rev2run.py $data 1 1 1 1 2 1 0 10 $k $n $ind

wait
# done
exit 0
'''

pbs_template = '''
#!/bin/bash -l
# declare a name for this job to be my_serial_job
# it is recommended that this name be kept to 16 characters or less
#PBS -N $algorithm-$data-$k-$n-$ind
#PBS -j oe
#PBS -l mem=$vf

# request the queue (enter the possible names, if omitted, default is the default)
# this job is going to use the default
#PBS -q default

# request 1 node
#PBS -l nodes=1:ppn=1

# request 0 hours and 15 minutes of wall time
# (Default is 1 hour without this directive)
#PBS -l walltime=$time

# mail is sent to you when the job starts and when it terminates or aborts 
####PBS -m bea

# specify your email address 
####PBS -M John.Smith@dartmouth.edu

# By default, PBS scripts execute in your home directory, not the
# directory from which they were submitted. The following line
# places the job in the directory from which the job was submitted.

module add python/3.6-GPU
source activate default
cd $HOME/research/fake-review/src

# run the program using the relative path

OUTPUT_DIR="../res/$algorithm/$data/"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

python algs/rev2run.py $data 1 1 1 1 1 1 0 10 $k $n $ind
python algs/rev2run.py $data 1 2 1 1 1 1 0 10 $k $n $ind
python algs/rev2run.py $data 1 1 2 1 1 1 0 10 $k $n $ind
python algs/rev2run.py $data 1 1 1 2 1 1 0 10 $k $n $ind
python algs/rev2run.py $data 1 1 1 1 2 1 0 10 $k $n $ind

exit 0
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='rev2run')
    parser.add_argument('-d', '--data', action='store', choices=['alpha', 'amazon', 'epinions', 'otc'], help='target dataset')
    parser.add_argument('-c', '--clean', action='store_true', help='clean up the qjobs and outputs')
    parser.add_argument('-a', '--alg', action='store', choices=['rev2'], default='rev2', help='algorithm')
    parser.add_argument('-t', '--template', action='store', choices=['pbs', 'anthill'], default='anthill',
                        help='pbs or anthill (sun grid engine)')
    parsed = parser.parse_args(sys.argv[1:])
    
    print(parsed)

    if parsed.clean:
        print('cleanup qjobs')
        proc_ret = subprocess.run('rm -vf *.qjob', shell=True)
        print(proc_ret)
        print('cleanup outputs')
        proc_ret = subprocess.run('rm -vf *.o*', shell=True)
        print(proc_ret)
        proc_ret = subprocess.run('rm -vf *.po*', shell=True)
        print(proc_ret)
        exit()
    
    n_range = list(range(0, 51, 5))
    n_range[0] = 1

    if not os.path.exists(parsed.alg):
        os.mkdir(parsed.alg)
    if not os.path.exists(os.path.join(parsed.alg, parsed.data)):
        os.mkdir(os.path.join(parsed.alg, parsed.data))

    template = anthill_template
    if parsed.template == 'pbs':
        template = pbs_template
    
    for k in range(10):
        for n in n_range:
            for ind in range(50):
                qjob_name = 'rev2run-%s-%d-%d-%d.qjob' %(parsed.data, k, n, ind)
                if parsed.data == 'epinions':
                    script = template.replace('$k', str(k)).replace('$n', str(n)).replace('$data', parsed.data).replace('$ind', str(ind)).replace('$vf', '20G')
                else:
                    script = template.replace('$k', str(k)).replace('$n', str(n)).replace('$data', parsed.data).replace('$ind', str(ind)).replace('$vf', '10G')
                with open(os.path.join(parsed.alg, parsed.data, qjob_name), 'w') as f:
                    f.write(script)

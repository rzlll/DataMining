#!/usr/bin/env python3

import sys, os
import argparse
import subprocess

template = '''
#!/usr/bin/env bash

# The name of the job, can be anything, simply used when displaying the list of running jobs
#$ -N rev2-$data-$k-$n-$ind
# Combining output/error messages into one file
#$ -j y

# Set memory request:
#$ -l vf=2G

# ironfs access
###$ -l ironfs

# number of processes (cores)
### -pe smp 8

# Set walltime request:
#$ -l h_rt=2:59:59

# One needs to tell the queue system to use the current directory as the working directory
# Or else the script may fail as it will execute in your top level home directory /home/username
#$ -cwd
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='rev2run')
    parser.add_argument('-d', '--data', action='store', choices=['alpha', 'amazon', 'epinions', 'otc'], help='target dataset')
    parser.add_argument('-c', '--clean', action='store_true', help='clean up the qjobs and outputs')
    parser.add_argument('-a', '--alg', action='store', choices=['rev2'], default='rev2', help='algorithm')
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
    
    for k in range(10):
        for n in n_range:
            for ind in range(50):
                qjob_name = 'rev2run-%s-%d-%d-%d.qjob' %(parsed.data, k, n, ind)
                script = template.replace('$k', str(k)).replace('$n', str(n)).replace('$data', parsed.data).replace('$ind', str(ind))
                with open(os.path.join(parsed.alg, parsed.data, qjob_name), 'w') as f:
                    f.write(script)
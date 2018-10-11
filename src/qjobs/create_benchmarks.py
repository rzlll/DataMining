#!/usr/bin/env python3

import sys, os, argparse, subprocess
import itertools

template = '''
#!/usr/bin/env bash

# The name of the job, can be anything, simply used when displaying the list of running jobs
#$ -N $algorithm-$data-$k-$n-$ind
# Combining output/error messages into one file
#$ -j y

# Set memory request:
#$ -l vf=5G

# ironfs access
###$ -l ironfs

# number of processes (cores)
### -pe smp 8

# Set walltime request:
#$ -l h_rt=0:59:59

# One needs to tell the queue system to use the current directory as the working directory
# Or else the script may fail as it will execute in your top level home directory /home/username
#$ -cwd
# then you tell it retain all environment variables (as the default is to scrub your environment)
####$ -V
# Now comes the command to be executed

# activate virtual env for python3

source $HOME/venv/bin/activate
cd $HOME/research/fake-review/src

OUTPUT_DIR="../res/$algorithm/$data/"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

python algs/$algorithm $data $k $n $ind

wait
# done
exit 0
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='create qjobs')
    parser.add_argument('-d', '--data', action='store', choices=['alpha', 'amazon', 'epinions', 'otc'], default='alpha', help='target dataset')
    parser.add_argument('-c', '--clean', action='store_true', help='clean up the qjobs and outputs')
    parser.add_argument('-a', '--alg', action='store', choices=['bn', 'feagle', 'fraudar', 'trust', 'rsd', 'bad'], default='bn', help='algorithm')
    parsed = parser.parse_args(sys.argv[1:])
    
    print(parsed)

    if parsed.clean:
        print('cleanup configs and qjobs')
        proc_ret = subprocess.run('rm -vf *.json', shell=True)
        print(proc_ret)
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

    for k in range(10):
        for n in n_range:
            for ind in range(50):
                qjob_name = '%s-%s-%d-%d-%d.qjob' %(parsed.alg, parsed.data, k, n, ind)
                script = template.replace('$data', parsed.data).replace('$k', str(k)).replace('$n', str(n)).replace('$algorithm', parsed.alg).replace('$ind', str(ind))
                with open(os.path.join(parsed.alg, qjob_name), 'w') as fp:
                    fp.write(script)
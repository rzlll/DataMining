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
#$ -l vf=$vf

# ironfs access
###$ -l ironfs

# number of processes (cores)
### -pe smp 8

# Set walltime request:
#$ -l h_rt=9:59:59

# One needs to tell the queue system to use the current directory as the working directory
# Or else the script may fail as it will execute in your top level home directory /home/username

# the cwd is somehow causing problems
####$ -cwd
#$ -wd /home/ifsdata/scratch/rliu/qlog/

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

python algs/$algorithm.py $data $k $n $ind

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
    if not os.path.exists(os.path.join(parsed.alg, parsed.data)):
        os.mkdir(os.path.join(parsed.alg, parsed.data))

    create_list = []
    skip_list = []
    for k in range(10):
        for n in n_range:
            for ind in range(50):
                qjob_name = '%s-%s-%d-%d-%d.qjob' %(parsed.alg, parsed.data, k, n, ind)
                target_path = '../../res/%s/%s/%s-%d-%d-%d.csv' %(parsed.alg, parsed.data, parsed.data, k, n, ind)
                if os.path.exists(target_path):
                    skip_list.append(target_path)
                    continue
                # epinions is large
                if parsed.data == 'epinions':
                    script = template.replace('$data', parsed.data).replace('$k', str(k)).replace('$n', str(n)).replace('$algorithm', parsed.alg).replace('$ind', str(ind)).replace('$vf', '20G')
                else:
                    script = template.replace('$data', parsed.data).replace('$k', str(k)).replace('$n', str(n)).replace('$algorithm', parsed.alg).replace('$ind', str(ind)).replace('$vf', '10G')

                with open(os.path.join(parsed.alg, parsed.data, qjob_name), 'w') as fp:
                    fp.write(script)
                create_list.append(target_path)
    print('skip', len(skip_list))
    print('show 10', skip_list[-10:])
    print('create', len(create_list))
    print('show 10', create_list[-10:])

#!/usr/bin/env python3

import sys, os, argparse, subprocess
import itertools

anthill_template = '''
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
#$ -l h_rt=$time

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

$cmd_template

wait
# done
exit 0
'''

cmd_template = 'python rtv-non-socks.py %s %d %d %d %d %d %d %d %d %d %d %d'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='create qjobs')
    parser.add_argument('-d', '--data', action='store', choices=['alpha', 'amazon', 'epinions', 'otc'], default='alpha', help='target dataset')
    parser.add_argument('-c', '--clean', action='store_true', help='clean up the qjobs and outputs')
    parser.add_argument('-a', '--alg', action='store', choices=['rtv'], default='rtv', help='algorithm')
    parser.add_argument('-t', '--template', action='store', choices=['anthill'], default='anthill', help='pbs or anthill (sun grid engine)')
    parser.add_argument('-p', '--produce', action='store_true', help='output or not')
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
    
    n_range = list(range(0, 51, 10))
    n_range[0] = 1

    if not os.path.exists(parsed.alg):
        os.mkdir(parsed.alg)
    if not os.path.exists(os.path.join(parsed.alg, parsed.data)):
        os.mkdir(os.path.join(parsed.alg, parsed.data))

    template = anthill_template
    if parsed.template == 'pbs':
        template = pbs_template
        
    a1_list = range(1, 3)
    a2_list = range(1, 3)

    b1_list = range(1, 3)
    b2_list = range(1, 3)

    g1_list = range(20, 100, 30)
    g2_list = range(10, 20, 5)
    g3_list = range(1, 10, 5)
    g4_list = range(1, 10, 5)
    
    maxiter = 30
    tnum = 100
    vnum = 500

    create_list = []
    skip_list = []
    for k in range(0, 10, 3):
        for n in n_range:
            for ind in range(0, 5):
                cmd_list = []
                for a1, a2 in itertools.product(a1_list, a2_list):
                    if a1 == a2 != 1:
                        continue
                    for b1, b2 in itertools.product(b1_list, b2_list):
                        if b1 == b2 != 1:
                            continue
                        for g1, g2, g3, g4 in itertools.product(g1_list, g2_list, g3_list, g4_list):
                            if g3 == g4 != 1:
                                continue
                            cmd = cmd_template % (parsed.data, a1, a2, b1, b2, g1, g2, g3, g4, maxiter, tnum, vnum)
                            cmd_list += [cmd]
                            target_path = '../../res/%s/%s/%s-%d-%d-%d-%d-%d-%d-%d-%d-%d-%d-%d.csv' % (parsed.alg, parsed.data, parsed.data, a1, a2, b1, b2, g1, g2, g3, g4, k, n, ind)
                            job_path = './%s/%s-%d-%d-%d-%d-%d-%d-%d-%d-%d-%d-%d.csv' % (parsed.alg, parsed.data, a1, a2, b1, b2, g1, g2, g3, g4, k, n, ind)

                            qjob_name = '%s-%s-%d-%d-%d.qjob' %(parsed.alg, parsed.data, k, n, ind)
                            if os.path.exists(target_path):
                                skip_list.append(target_path)
                                continue
                            # epinions is large and needs a lot of time to produce
                            if parsed.data == 'epinions':
                                script = template.replace('$data', parsed.data).replace('$k', str(k)).replace('$n', str(n)).replace('$algorithm', parsed.alg).replace('$ind', str(ind)).replace('$vf', '20G').replace('$time', '47:59:59')
                            else:
                                script = template.replace('$data', parsed.data).replace('$k', str(k)).replace('$n', str(n)).replace('$algorithm', parsed.alg).replace('$ind', str(ind)).replace('$vf', '10G').replace('$time', '23:59:59')

                            script = script.replace('$cmd_template', cmd)
                            if parsed.produce:
                                with open(job_path, 'w') as fp:
                                    fp.write(script)
                            create_list.append(target_path)
    print('skip', len(skip_list))
    print('show 10', skip_list[-10:])
    print('create', len(create_list))
    print('show 10', create_list[-10:])

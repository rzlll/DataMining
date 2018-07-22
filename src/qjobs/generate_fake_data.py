#!/usr/bin/env python3

sample='''
#!/usr/bin/env bash
# Generate fake data

# The name of the job, can be anything, simply used when displaying the list of running jobs
#$ -N fake-$data-$k-$n
# Combining output/error messages into one file
#$ -j y
# Set memory request:
#$ -l vf=2G
# Set walltime request:
#$ -l h_rt=0:29:59
# One needs to tell the queue system to use the current directory as the working directory
# Or else the script may fail as it will execute in your top level home directory /home/username
#$ -cwd
# then you tell it retain all environment variables (as the default is to scrub your environment)
#$ -V
# Now comes the command to be executed
source $HOME/venv/bin/activate

# for data in alpha amazon epinions otc; do
python ./fake.py $data $k $n
# done

exit 0
'''

for d in ['alpla', 'amazon', 'epinions', 'otc']:
    for k in range(11):
        for n in range(11):
            tmp = sample.replace('$k', str(k)).replace('$n', str(n)).replace('$data', d)
            with open('fake-%s-%d-%d.qjob', 'w') as f:
                f.write(tmp)

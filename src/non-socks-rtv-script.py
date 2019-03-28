#!/usr/bin/env python3

import sys, os, argparse
import itertools

parser = argparse.ArgumentParser(description='non socks rtv')
parser.add_argument('-d', '--data', type=str, default='alpha', choices=['alpha', 'amazon', 'epinions', 'otc'], help='data name', required=True)
parser.add_argument('-t', '--tnum', type=int, default=100, help='trusted size')
parser.add_argument('-v', '--vnum', type=int, default=500, help='verified size')
parsed = parser.parse_args(sys.argv[1:])

a1_list = range(1, 4)
a2_list = range(1, 4)

b1_list = range(1, 4)
b2_list = range(1, 4)

g1_list = range(20, 100, 20)
g2_list = range(10, 20, 5)
g3_list = range(1, 10, 5)
g4_list = range(1, 10, 5)

template = 'python rtv-non-socks.py %s %d %d %d %d %d %d %d %d %d %d %d'
maxiter = 30
dname = parsed.data
tnum = parsed.tnum
vnum = parsed.vnum

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
            cmd_list += [template % (dname, a1, a2, b1, b2, g1, g2, g3, g4, maxiter, tnum, vnum)]
print(len(cmd_list))
print(cmd_list[:10])

parallel_num = 20
cmd_parallel = [' &\n'.join(cmd_list[i:i+parallel_num]) + ' &' for i in range(0, len(cmd_list), parallel_num)]

cmd_script = '#!/usr/bin/env bash\n' + '\nwait\n'.join(cmd_parallel) + '\nwait'

print(cmd_script)

with open('non-socks-rtv.sh', 'w') as fp:
    fp.write(cmd_script)

#!/usr/bin/env bash

dname=$1

python algs-non-socks/bad.py $dname &
python algs-non-socks/bn.py $dname &
python algs-non-socks/feagle.py $dname &
python algs-non-socks/fraudar.py $dname &
python algs-non-socks/rsd.py $dname &
python algs-non-socks/trust.py $dname &

python algs-non-socks/rev2run.py $dname 2 1 1 1 1 1 1 20 &
python algs-non-socks/rev2run.py $dname 1 2 1 1 1 1 1 20 &
python algs-non-socks/rev2run.py $dname 1 1 2 1 1 1 1 20 &
python algs-non-socks/rev2run.py $dname 1 1 1 2 1 1 1 20 &
python algs-non-socks/rev2run.py $dname 1 1 1 1 1 1 1 20 &

python rtv-non-socks.py $dname 2 1 1 1 10 2 1 1 20 &
python rtv-non-socks.py $dname 1 2 1 1 10 2 1 1 20 &
python rtv-non-socks.py $dname 1 1 2 1 10 2 1 1 20 &
python rtv-non-socks.py $dname 1 1 1 2 10 2 1 1 20 &
python rtv-non-socks.py $dname 1 1 1 1 10 2 1 1 20 &

wait

#!/usr/bin/env bash

dname=$1
tnum=$2
vnum=$3

python rtv-non-socks.py $dname 2 1 1 1 10 2 1 1 30 $tnum $vnum &
python rtv-non-socks.py $dname 1 2 1 1 10 2 1 1 30 $tnum $vnum &
python rtv-non-socks.py $dname 1 1 2 1 10 2 1 1 30 $tnum $vnum &
python rtv-non-socks.py $dname 1 1 1 2 10 2 1 1 30 $tnum $vnum &
python rtv-non-socks.py $dname 1 1 1 1 10 2 1 1 30 $tnum $vnum &

wait

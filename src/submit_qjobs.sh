#!/usr/bin/env sh

for qjob in $@
do
    qsub $qjob
done
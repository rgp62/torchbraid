#!/bin/bash

for CASE in 1 2 3 4 5 6 7 8
do
    for CPUS in 1 2 4 8 16
    do
        echo $CPUS
        mpirun -np $CPUS python main.py --batch-size=64 --steps=512 --channels=64 --force-lp &> strong_${CPUS}_${CASE}.txt
    done
done

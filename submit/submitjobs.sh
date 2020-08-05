#!/bin/bash

jobname=$1

addqueue -O -q gpulong -n 1x4 -e -s -m 3 -o /users/exet4487/launchpad/${jobname}_head.txt testhead.sh $jobname
sleep 40
cp /users/exet4487/getout_glamdring/temp_model.py /users/exet4487/hyperopt_job/temp_model.py
addqueue -O -q gpulong -n 1x4 -s -m 3 -o /users/exet4487/launchpad/${jobname}_1.txt testlaunch.sh $jobname
addqueue -O -q gpulong -n 1x8 -s -m 3 -o /users/exet4487/launchpad/${jobname}_2.txt testlaunch.sh $jobname
addqueue -O -q gpulong -n 1x8 -s -m 3 -o /users/exet4487/launchpad/${jobname}_3.txt testlaunch.sh $jobname
addqueue -O -q gpulong -n 1x8 -s -m 3 -o /users/exet4487/launchpad/${jobname}_4.txt testlaunch.sh $jobname
addqueue -O -q gpulong -n 1x4 -s -m 3 -o /users/exet4487/launchpad/${jobname}_5.txt testlaunch.sh $jobname



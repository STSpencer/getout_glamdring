#!/bin/bash
runname="$1"
#module load python
source /users/exet4487/launchpad/venv/bin/activate
LD_LIBRARY_PATH="/usr/local/shared/cuda/cuda-10.1/lib64"
#python3 /users/exet4487/getout_glamdring/hypertrain_init.py
export PYTHONPATH=/users/exet4487/getout_glamdring/
cd /users/exet4487/hyperopt_job
hyperopt-mongo-worker --mongo="mongo://exet4487:admin123@192.168.0.200:27017/jobs" --exp-key=$runname --workdir='/users/exet4487/hyperopt_job/' --poll-interval=0.1

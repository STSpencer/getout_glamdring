#!/bin/bash
runname="$1"
#module load python
source /users/exet4487/launchpad/venv/bin/activate
LD_LIBRARY_PATH="/usr/local/shared/cuda/cuda-10.1/lib64"
#python3 /users/exet4487/getout_glamdring/hypertrain_init.py
cd /users/exet4487/getout_glamdring/
python3 /users/exet4487/getout_glamdring/hypertrain.py $runname

#!/bin/sh
indir=$1
model=$2
nfeat=$3
niter=$4

# activate the virtual environment
. code/localpython/venv/python37/bin/activate

# call the script
./code/predict_ppa.py -indir $indir -model $model -nfeat $nfeat -niter $niter

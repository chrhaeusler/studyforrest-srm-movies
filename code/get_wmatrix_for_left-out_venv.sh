#!/bin/sh
indir=$1
sub=$2
model=$3
nfeat=$4
niter=$5

# activate the virtual environment
. ~/python/environments/datalad/bin/activate

which python
pwd

# call the script
./code/get_wmatrix_for_left-out.py -indir $indir -sub $sub -model $model -nfeat $nfeat -niter $niter

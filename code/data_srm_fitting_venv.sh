#!/bin/sh
sub=$1
outdir=$2
nfeat=$3
niter=$4

# activate the virtual environment
. ~/python/environments/datalad/bin/activate

which python
pwd

# call the script
./code/data_srm_fitting.py -sub $sub -outdir $outdir -nfeat $nfeat -niter $niter

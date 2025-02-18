#!/bin/sh
sub=$1
outdir=$2
nfeat=$3
niter=$4
rseed=$5

# activate the virtual environment
. code/localpython/venv/python37/bin/activate

# call the script
./code/data_srm_fitting.py -sub $sub -outdir $outdir -nfeat $nfeat -niter $niter -rseed $rseed

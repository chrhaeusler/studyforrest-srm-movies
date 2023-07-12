#!/bin/sh
indir=$1
model=$2
nfeat=$3
niter=$4

# activate the virtual environment
. code/localpython/venv/python37/bin/activate

# FSL v5.0.9 directory
FSLDIR=~/fsl
. ${FSLDIR}/etc/fslconf/fsl.sh
PATH=${FSLDIR}/bin:${PATH}
export FSLDIR PATH
. ${FSLDIR}/etc/fslconf/fsl.sh

# call the script
./code/predict_ppa.py -indir $indir -model $model -nfeat $nfeat -niter $niter

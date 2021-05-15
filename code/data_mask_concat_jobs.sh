#!/bin/bash

##### Datalad Handbook 3.2 ######
##### DataLad-centric analysis with job scheduling and parallel computing
# http://handbook.datalad.org/en/latest/beyond_basics/101-170-dataladrun.html

# top-level analysis dataset with subdatasets
# $ datalad create parallel_analysis
# $ cd parallel_analysis
# a) pipeline dataset (with a configured software container) 
# $ datalad clone -d . https://github.com/ReproNim/containers.git
# b) input dataset
# $ datalad clone -d . /path/to/my/rawdata
# data analysis with software container that performs a set of analyses
# results will be aggregated into a top-level dataset 

# Individual jobs are computed in throw-away dataset clones (& branches) 
# to avoid unwanted interactions between parallel jobs.

# results are pushed back (as branches) to into the target dataset.

# A manual merge aggregates all results into the master branch of the dataset.

# following analysis processes rawdata with 
# - a pipeline from 
# - collects outcomes in toplevel parallel_analysis dataset 


# You could also add and configure the container using datalad containers-add 
# to the top-most dataset. This solution makes the container less usable, though.
# If you have more than one application for a container, keeping it as a 
# standalone dataset can guarantee easier reuse. 

# what you will submit as a job with a job scheduler
# is  a shell script that contains all relevant data analysis steps
# and not a datalad containers-run call

# but datalad run does not support concurrent execution in the same dataset clone.
# Solution: create one throw-away dataset clone for each job.

# We treat cluster compute nodes like contributors to the analyses: 
# They clone the analysis dataset hierarchy into a temporary location,
# run the computation, push the results, and remove their temporary dataset again

# The compute job clones the dataset to a unique place, so that it can run a 
# containers-run command inside it without interfering with any other job. 

# fail whenever something is fishy, use -x to get verbose logfiles
set -e -u -x

# we pass arbitrary arguments via job scheduler and can use them as variables
indir=$1

# The first part of the script is therefore to navigate to a unique location, 
# and clone the analysis dataset to it.
# go into unique location
cd /tmp
# clone the analysis dataset. flock makes sure that this does not interfere
# with another job finishing and pushing results back at the same time
flock --verbose $DSLOCKFILE datalad clone /data/group/psyinf/studyforrest-srm-movies chaeusler-concat

cd chaeusler-concat
# This dataset clone is temporary: It will exist over the course of one analysis/job only, 
# but before it is being purged, all of the results it computed will be pushed 
# to the original dataset. This requires a safe-guard: If the original dataset 
# receives the results from the dataset clone, it knows about the clone and its 
# state. In order to protect the results from someone accidentally synchronizing 
# (updating) the dataset from its linked dataset after is has been deleted, 
# the clone should be created as a “trow-away clone” right from the start. By 
# running git annex dead here, git-annex disregards the clone, preventing the 
# deletion of data in the clone to affect the original dataset.
# announce the clone to be temporary
git annex dead here

# The datalad push to the original clone location of a dataset needs to be prepared 
# carefully. The job computes one result (out of of many results) and saves it, 
# thus creating new data and a new entry with the run-record in the dataset 
# history. But each job is unaware of the results and commits produced by other 
# branches. Should all jobs push back the results to the original place (the 
# master branch of the original dataset), the individual jobs would conflict with 
# each other or, worse, overwrite each other (if you don’t have the default 
# push configuration of Git).

# The general procedure and standard Git workflow for collaboration, therefore, 
# is to create a change on a different, unique branch, push this different 
# branch, and integrate the changes into the original master branch via a merge 
# in the original dataset4.

# In order to do this, prior to executing the analysis, the script will checkout 
# a unique new branch in the analysis dataset. The most convenient name for the 
# branch is the Job-ID, an identifier under which the job scheduler runs an 
# individual job. This makes it easy to associate a result (via its branch) 
# with the log, error, or output files that the job scheduler produces5, and 
# the real-life example will demonstrate these advantages more concretely.

# git checkout -b <name> creates a new branch and checks it out
# checkout a unique branch
git checkout -b "job-$JOBID"

# $JOB-ID isn’t hardcoded into the script but it can be given to the script as 
# an environment or input variable at the time of job submission. 

# Next, its time for the containers-run command. The invocation will depend on 
# the container and dataset configuration (both of which are demonstrated in the 
# real-life example in the next section), and below, we pretend that the 
# container invocation only needs an input file and an output file. These input 
# file is specified via a bash variables ($inputfile) that will be defined in 
# the script and provided at the time of job submission via command line argument
# from the job scheduler, and the output file name is based on the input file name.

# After the containers-run execution in the script, the results can be pushed back to the dataset sibling origin6:

# run the job
datalad run \
-m "Computing data of subject ${indir}" \
--explicit \
--input $indir \
--output $indir \
./code/data_mask_concat_runs.py \
-sub "{inputs}" -outdir "{outputs}"
# push, with filelocking as a safe-guard
flock --verbose $DSLOCKFILE datalad push --to origin

# Done - job handler should clean up workspace

# manually merge it 
# git merge -m "Merge results from job cluster XY" $(git branch -l | grep 'job-' | tr -d ' ')
# delete branches matching a pattern
# git branch | grep "job-*" | xargs git branch -D

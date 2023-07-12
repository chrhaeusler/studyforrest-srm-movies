# Assessing the quantity of data for functional alignment to estimate responses in the "parahippocampal place area": from raw data to results

[![made-with-datalad](https://www.datalad.org/badges/made_with.svg)](https://datalad.org)

This repository contains the raw data and all code to generate the results in
Chapter 5 of the PhD thesis "Exploring naturalistic stimulus paradigms as an alternative to a
task-based functional localizer paradigm" written by Häusler, C.O.

If you have never used [DataLad](https://www.datalad.org/) before, please read the
section on DataLad datasets below.



## DataLad datasets and how to use them

This repository is a [DataLad](https://www.datalad.org/) dataset. It allows
fine-grained data access up to the level of single files.  In order to use this
repository for data retrieval, [DataLad](https://www.datalad.org/) is required.
It is a free and open source command line tool, available for all major
operating systems, and builds up on Git and
[git-annex](https://git-annex.branchable.com/) to allow sharing, synchronizing,
and version controlling collections of large files. You can find information on
how to install DataLad at
[handbook.datalad.org/en/latest/intro/installation.html](http://handbook.datalad.org/en/latest/intro/installation.html).


### Get the dataset

A DataLad dataset can be `cloned` by running

```
datalad clone <url>
```
Once a dataset is cloned, it is a light-weight directory on your local machine.
At
this point,
it contains only small metadata and information on the identity of the files in the dataset,
but not actual *content* of the (sometimes large) data files.


### Retrieve dataset content

After cloning a dataset, you can retrieve file contents by running
```
datalad get <path/to/directory/or/file>
```
This command will trigger a download of the files, directories, or subdatasets you have specified.

DataLad datasets can contain other datasets, so called *subdatasets*. If you clone the top-level
dataset, subdatasets do not yet contain metadata and information on the identity of files,
but appear to be empty directories. In order to retrieve file availability metadata in
subdatasets, run

```
datalad get -n <path/to/subdataset>
```
Afterwards, you can browse the retrieved metadata to find out about subdataset contents, and
retrieve individual files with `datalad get`. If you use `datalad get <path/to/subdataset>`,
all contents of the subdataset will be downloaded at once.


### Stay up-to-date

DataLad datasets can be updated. The command `datalad update` will *fetch* updates and store them
on a different branch (by default `remotes/origin/master`). Running
```
datalad update --merge
```
will *pull* available updates and integrate them in one go.


### More information

More information on DataLad and how to use it can be found in the DataLad Handbook at
[handbook.datalad.org](http://handbook.datalad.org/en/latest/index.html). The chapter
"DataLad datasets" can help you to familiarize yourself with the concept of a dataset.



## Dataset structure and contents

- `code/`: a local installation of Python 3.7 and all custom code
	- `localpython/python37`: the local installation of Python 3.7.17
	- `localpython/venv/python37`: a virtual environment comprising the packages necessary to run the calculations
	- `*.py`: Python 3.7 scripts
	- `*.sh`: Bash scripts
	- `*.submit`: submit files for HTCondor that distributes calculations across a computer cluster
- `inputs/`: building blocks from other sources; DataLad datasets installed as subdatasets
	- `studyforrest-data-templatetransforms`: participant/scan-specific template images
	and transformations between these respective image spaces 
	(cf. [repo on github](https://github.com/psychoinformatics-de/studyforrest-data-templatetransforms/))
	- `studyforrest-data-visualrois`: data and results of 
	[Sengupta, A. et al. (2016)](https://github.com/psychoinformatics-de/studyforrest-data-visualrois.git)
	- `studyforrest-ppa-analysis`: data and results of 
	[Häusler, Eickhoff, & Hanke (2022)](https://gin.g-node.org/chaeusler/studyforrest-ppa-analysis)
- `masks/`: group masks and atlases
- `sub-*/`: individual subject folders that contain 
	- `masks/`: masks in the corresponding subject's voxel space (e.g., PPA and field of view)
	- `sub-*_task-*_run-*_bold_filtered.nii.gz`: time series that were used as input in FSL in Sengupta et al. (2016), and Häusler, Eickhoff, & Hanke (2022)
	- `sub-*_task_*_run-*-*_bold-filtered.npy`: masked time series, z-scored per paradigm/run
 	- `sub-*_ao-av-vis_concatenated_zscored.npy`: all paradigms concatenated and z-scored
	- `models/`: the shared response models calculated from the training subjects' data
	- `matrices/`: the transformation matrices of the subject obtained from aligning her/him to a shared response model
	- `predictions/`: the predicted Z-maps
- `results/`: final statistics and figures
	- `statistics_cronbachs.csv`: Cronbach's Alpha of the empirical Z-maps
	- `corr_*-ppa-vs-estimation_srm-ao-av-vis_feat*.csv`: correlations between empirical and predicted Z-maps
	- `statistics_t-tests.csv`: results of the t-tests


	
## Cookbook -- How reproduce this dataset from scratch


### Setting up variables and the virtual environment

	# the present working directory
	mainDir=$PWD
	# activate the virtual environment
	source code/localpython/venv/python37/bin/activate
	# FSL v5.0.9 directory
	FSLDIR=~/fsl
	. ${FSLDIR}/etc/fslconf/fsl.sh
	PATH=${FSLDIR}/bin:${PATH}
	export FSLDIR PATH
	. ${FSLDIR}/etc/fslconf/fsl.sh
	
	
### Reproduce results of Sengupta et al. (2016)
	# install the dataset
	datalad install -d . -s https://github.com/psychoinformatics-de/studyforrest-data-visualrois.git inputs/studyforrest-data-visualroi
	
	# create union of visual ROIs
	# manually add 'create-roi-overlaps.sh'
	datalad save -m 'add script to create union of visual rois (Sengupta et al., 2016)'
	# run it; outputs result to 'masks/in_mni'
	datalad run -m 'create union of visual rois' ./code/create-roi-overlaps.sh
	
	# retrieve file availability metadata of the subdataset containing the time series data
	datalad get -n inputs/studyforrest-data-visualrois/src/aligned
	# get the data
	datalad get inputs/studyforrest-data-visualrois/src/aligned/sub-*/in_bold3Tp2/sub-*_task-objectcategories_run-*_bold.nii.gz

	# rerun first-level analysis	
	# get the FSL onset files for each subject
	datalad get inputs/studyforrest-data-visualrois/sub-*/onsets/run-*/*.txt
	# go into the dataset's directory
	cd inputs/studyforrest-data-visualrois
	# manually adjust 'code/despike.submit' and save
	datalad save -m 'adjust (absolute) paths in code/despike.submit'
	# run the despiking on a computer cluster
	condor_submit code/despike.submit
	# save results
	datalad save -m 'despike fMRI data'
	# manually adjust paths to current environment (FSL seems to require absolute paths) 
	datalad save -m 'adjust (absolute) paths 1stlevel_design.fsf'
	datalad save -m 'adjust (absolute) paths generate_1st_level_design.sh'
	# generate first-level design files
	./code/generate_1st_level_design.sh
	# run first-level analyses on a computer cluster
	# outputs results to 'inputs/studyforrest-data-visualrois/sub-*/run-*.feat''
	condor_submit code/compute_1stlvl_glm.submit
	# save results
	datalad save -m '1st lvl results'
	# back to main directory
	cd $mainDir


### Reproduce results of Häusler, Eickhoff, & Hanke (2022)
	# install the dataset
	datalad install -d . -s https://gin.g-node.org/chaeusler/studyforrest-ppa-analysis inputs/studyforrest-ppa-analysis
	
	# retrieve file availability metadata of the subdataset containing the time series data
	datalad get -n inputs/studyforrest-ppa-analysis/inputs/studyforrest-data-aligned
	# get the data
	datalad get inputs/studyforrest-ppa-analysis/inputs/studyforrest-data-aligned/sub-??/in_bold3Tp2/sub-??_task-a?movie_run-?_bold*.*
	
	# retrieve file availability metadata of the subdataset containing the motion correction parameters of the audio-description
	datalad get -n inputs/studyforrest-ppa-analysis/inputs/phase1
	# get the correction parameters
	datalad get inputs/studyforrest-ppa-analysis/inputs/phase1/sub???/BOLD/task001_run00?/bold_dico_moco.txt
	
	# retrieve file availability metadata of the subdataset containing templates and transforms
	datalad get -n inputs/studyforrest-ppa-analysis/inputs/studyforrest-data-templatetransforms
	# get the actual data
	datalad get inputs/studyforrest-ppa-analysis/inputs/studyforrest-data-templatetransforms/sub-*/bold3Tp2/
	datalad get inputs/studyforrest-ppa-analysis/inputs/studyforrest-data-templatetransforms/templates/*
	
	# rerun first-level analysis
	# go into the dataset's directory
	cd inputs/studyforrest-ppa-analysis
	# manually adjust paths to current environment (FSL seems to require absolute paths)
	datalad save -m 'adjust paths in 1st lvl FEAT design files (movie & group, individuals)'
	# run first-level analyses on a computer cluster
	# outputs results to 
	# inputs/studyforrest-ppa-analysis/sub-*/run-*_audio-ppa-grp.feat, and
	# inputs/studyforrest-ppa-analysis/sub-*/run-*_movie-ppa-grp.feat
	condor_submit code/compute_1st-lvl_movie-ppa-ind.submit
	condor_submit code/compute_1st-lvl_audio-ppa-ind.submit
	# save results
	datalad save -m '1st results movie & audio (individuals)'
	
	# rerun second-level analysis
	# create templates
	./code/reg2std4feat inputs/studyforrest-data-templatetransforms bold3Tp2 bold3Tp2 sub-*/run-?_movie-ppa-ind.feat\
	./code/reg2std4feat inputs/studyforrest-data-templatetransforms bold3Tp2 bold3Tp2 sub-*/run-?_audio-ppa-ind.feat\
	# save it
	datalad save -m 'add templates & transformation matrices to 1st lvl result directories of Feat'
	# manually adjust paths to current environment (FSL seems to require absolute paths)
	datalad save -m 'adjust (absolute) paths in generate_2nd-lvl-design_*-ppa-ind.sh'
	datalad save -m 'adjust (absolute) paths in 2nd-lvl_*-ppa-ind.fsf'
	# from template to individual design files (audio, individuals); adjusted paths
	datalad rerun c0ada988773
	# from template to individual design files (movie, individuals); adjusted paths
	datalad rerun 9ece407eb20
	# run the second-level analyses on a computer cluster
	# outputs results to:
	# inputs/studyforrest-ppa-analysis/sub-*/2nd-lvl_audio-ppa-ind.gfeat
	# inputs/studyforrest-ppa-analysis/sub-*/2nd-lvl_movie-ppa-ind.gfeat
	# inputs/studyforrest-ppa-analysis/sub-*/2nd-lvl_audio-ppa-grp.gfeat
	# inputs/studyforrest-ppa-analysis/sub-*/2nd-lvl_movie-ppa-grp.gfeat
	condor_submit code/compute_2nd-lvl_audio-ppa-ind.submit
	condor_submit code/compute_2nd-lvl_movie-ppa-ind.submit
	condor_submit code/compute_2nd-lvl_audio-ppa-grp.submit
	condor_submit code/compute_2nd-lvl_movie-ppa-grp.submit
	# save results
	datalad save -m '2nd lvl results movie & audio'
	# back to main directory
	cd $mainDir


### Masks & probabilistic ROIs
	# add probabilistic ROIs created in fsleyes to 'masks/in_mni' manually
	datalad save -m 'add probabilistic ROIs extracted from fsleyes (Harvard-Oxford & MNI Prob Atlas)'
	
	# add field of view of the audio-description ('fov_tmpl_0.5.nii.gz') to 'masks/in_mni' manually
	datalad save -m 'add AO study FoV mask (in MNI152 space)'

	# warp masks from MNI152 into subject spaces
	# manually add the script 'masks-from-mni-to-bold3Tp2.py' that will call FSL's 'applywarp' command
	datalad save -m 'add masks-from-mni-to-bold3Tp2.py'
	# do the warping from MNI152 into subjects spaces
	# inputs from 'studyforrest-data-templatetransforms' are downloaded by the script
	# outputs results to 'sub-*/masks/in_bold3Tp2'
	datalad run -m 'warp MNI masks into individual bold3Tp2 spaces' \
	./code/masks-from-mni-to-bold3Tp2.py

    # manually add script that aligns t1w images with bold3Tp2
	datalad save -m 'add masks-from-t1w-to-bold3Tp2.py'
	# run it; outputs results to 'sub-*/masks/in_bold3Tp2'
	datalad run -m 'transform masks in t1w to individual bold3Tp2' \
	./code/masks-from-t1w-to-bold3Tp2.py

    # create binary AO FoV from 4D data, and binary gray matter masks for each subject
    # manually add script the script 
    datalad save -m 'add mask-builder-voxel-counter.py'
	# run it; outputs results to 'sub-*/masks/in_bold3Tp2'
	datalad run -m 'create individual AO FoV and gray matter masks' \
	./code/masks-builder-voxel-counter.py


### Reproduce time series of Sengupta et al. (2016) and Häusler, Eickhoff, & Hanke (2022)
	# FSL does not save a grand mean scaled version of the time series, so let's do it 
    # manually add 'grand_mean_for_4d.py' and 'grand_mean_for_4d.submit'
	datalad save -m 'add scripts that apply grand mean scaling to filtered functional data per subj & run'
	# run it on a computer cluster
	# outputs results to 'sub-*/sub-*_task-*_run-*_bold_filtered.nii.gz
	condor_submit code/grand_mean_for_4d.submit
	# save results
	datalad save -m 'save grand mean scaled runs (AV, AO, VIS) per subject'


### Preprocessing for the SRM
	# mask, z-score, and concat the time series
	# manually add 'data_mask_concat_runs.py', 'data_mask_concat_jobs.sh', and 'data_mask_concat_runs.submit'
	datalad save -m 'add scripts that mask & concat 4D data using HTcondor'
	# run it on a computer cluster
	# outputs results to:
	# 'sub-*/sub-*_task_aomovie-avmovie_run-1-8_bold-filtered.npy'
	# 'sub-*/sub-*_task_visloc_run-1-4_bold-filtered.npy'
	# 'sub-*/sub-*_ao-av-vis_concatenated_zscored.npy
	condor_submit code/data_mask_concat_runs.submit
	# save output
	datalad save -m 'save masked (individual gray matter & FoV) & concatenated runs'
	# manually merge branches it 
	git merge -m "Merge results from job cluster" $(git branch -l | grep 'job-' | tr -d ' ')
	# delete branches matching a pattern
	branch | grep "job-*" | xargs git branch -D


### Perform the model fit on training subjects' data
	# manually add scripts that calculate the common functional space and training subjects' transformations
	# ('data_srm_fitting.py', 'data_srm_fitting_venv.sh', 'data_srm_fitting.submit' )
	datalad save -m 'add scripts to perform SRM fitting (.py, .sh, .sh)'
	# run it on a computer cluster
	# outputs results to 'sub-*/models'
	condor_submit code/data_srm_fitting.submit 
	# save it
	datalad save -m 'save SRM models (computed on a cluster)'

### Obtain transformation matrices of the left-out subjects 
	# manually add script that calculate the transformation matrices
	datalad save -m 'add template of get_wmatrix_for_left-out.py'
	# manually add scripts that allow calculations on a cluster 
	# ('get_wmatrix_for_left-out_venv.sh', 'get_wmatrix_for_left-out.submit')
	datalad save -m 'add scripts (.submit, .sh) that allow calculation matrices on a cluster'
	# run it on a computer cluster
	# outputs results to 'sub-*/matrices'
	condor_submit code/get_wmatrix_for_left-out.submit
	# save output
	datalad save -m 'save test subjects matrices (computed on a cluster)'

### Predict PPA
	# manually add scripts that maps individual results through common spaces 
	# into a test subject's voxels space to predict his/her/whatever PPA;
	# also warps ROIs of training subjects via FSL's applywarp into voxel space 
	# of the test subject, and calculates the correlations between empirical 
	# and predicted Z-maps
	datalad save -m 'add predict_ppa.py'
	# manually add a HTCondor submit file
	datalad save -m 'add submit file predict_ppa.submit'
	# run it on a computer cluster
	# outputs results to:
	#  masks/in_mni/sub-*_*-PPA.nii.gz,
	# 'sub-*/masks/in_bold3Tp2'
	# 'sub-*/predictions', and
	# 'results/corr_*-ppa-vs-estimation_srm-ao-av-vis_feat10.csv'
	condor_submit code/predict_ppa.submit
	datalad save -m 'save predicted z-maps (computed on a cluster)'


### Calculate statistics
	# statistics_cronbachs.py
	datalad save -m 'add script that calculates cronbachs alpha for VIS runs, and AV & AO segments'
	# run it; outputs results to 'results/statistics_cronbachs.py
	datalad run -m 'calculate Cronbachs Alpha of empirical Z-maps' \
	./code/statistics_cronbachs.py

	# statistics_t-test-correlations.py
	datalad save -m 'add scripts for t-test of correlations'
	# run it; outputs results to 'results/statistics_t-tests.csv'
	datalad run -m 'calculate t-tests' ./code/statistics_t-test-correlations.py 

### ----- from here on, everything is automatized but not yet run via datalad run ------

### Create Plots
	# add plot_corr-of-glm-and-srm.py
	datalad save -m 'add plot_corr-of-glm-and-srm.py'
	%run code/plot_corr-of-glm-and-srm.py -model 'sub-01/models/srm-ao-av-vis_feat10-iter30.npz' -o results
	%run code/plot_corr-of-glm-and-srm.py -model 'sub-01/models/srm-ao-av-vis_shuffled-within_feat10-iter30-0001.npz' -o results
	%run code/plot_corr-of-glm-and-srm.py -model 'sub-01/models/srm-ao-av-vis_shuffled-across_feat10-iter30-0001.npz' -o results

	# plot_cronbachs.py
	datalad save -m 'add script for plotting stripplot Cronbachs'
	%run code/plot_cronbachs.py

	# plot_corr-emp-vs-estimation.py
	datalad save -m 'add script for plotting stripplot of correlations'

	%run code/plot_corr-emp-vs-estimation.py \
	-invis './results/corr_vis-ppa-vs-estimation_srm-ao-av-vis_feat10.csv' \
	-inav './results/corr_av-ppa-vs-estimation_srm-ao-av-vis_feat10.csv' \
	-inao './results/corr_ao-ppa-vs-estimation_srm-ao-av-vis_feat10.csv'

	# plot_voxel-counts.py
	datalad save -m 'add script for plotting voxels per subject-specific mask'
	%run code/plot_voxel-counts.py


<!---
### non-finalyzed scripts

### plot_srm.py
	# manually add script that ...
	datalad save -m 'add plot_srm.py'

### plot_bland-altman.py
	# manually add script that ...
	datalad save -m 'add script that plots Bland-Altman-Plots'



# From PPA study; do similarly according to current dataset

### comment: some cleaning that we did

In order to limit the dataset to an appropriate size, we dropped some files that were generated by FEAT during an intermediate stage of the first level analyses. More specifically, we dropped *filtered_func_data.nii.gz* (4D fMRI data after all filtering) and *res4d.nii.gz* (residual noise images) for every subject and run using the following commands 

    git annex unused
    git annex dropunused all --force
    datalad drop --nocheck sub*/*.feat/filtered_func_data.nii.gz
    datalad drop --nocheck sub*/*.feat/stats/res4d.nii.gz
    git rm sub-*/run-*.feat/filtered_func_data.nii.gz
    git rm sub-*/run-*.feat/stats/res4d.nii.gz
    
If necessary, the files can be obtained by rerunning the corresponding first level analysis.
-->

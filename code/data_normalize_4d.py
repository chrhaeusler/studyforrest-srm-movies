#!/usr/bin/env python3
'''
created on Mon March 29th 2021
author: Christian Olaf Haeusler
'''


import argparse
from glob import glob
import nibabel as nib
import numpy as np
import os
import re
import subprocess
from sklearn import preprocessing

# constants
VIS_FILTERED_PATTERN = 'inputs/studyforrest-data-visualrois/' + \
    'sub-??/run-?.feat/filtered_func_data.nii.gz'

AO_FILTERED_PATTERN = 'inputs/studyforrest-ppa-analysis/' + \
    'sub-??/run-?_audio-ppa-ind.feat/filtered_func_data.nii.gz'

AV_FILTERED_PATTERN = 'inputs/studyforrest-ppa-analysis/' + \
    'sub-??/run-?_movie-ppa-ind.feat/filtered_func_data.nii.gz'


def find_files(pattern):
    '''
    '''
    def sort_nicely(l):
        '''Sorts a given list in the way that humans expect
        '''
        convert = lambda text: int(text) if text.isdigit() else text
        alphanum_key = lambda key: [convert(c) for c in re.split(
            '([0-9]+)', key)]

        l.sort(key=alphanum_key)

        return l

    found_files = glob(pattern)
    found_files = sort_nicely(found_files)

    return found_files


def grand_mean_for_4d_file(in_fpath, stimulus='aomovie'):
    '''
    '''
    print("loading", in_fpath)

    if not os.path.exists(in_fpath):
        subprocess.call(['datalad', 'get', in_fpath])

    img = nib.load(in_fpath)
    img_data = img.get_fdata()

    # grand mean scaling for 4D data (e.g. in FSL):
    # An analysis step in which the voxel values in every
    # image are divided by the average global mean intensity of the whole
    # session. This effectively removes any mean global differences in
    # intensity between sessions.

    # flatten the data
    flat_data = np.matrix.flatten(img_data)
    scaled_data = flat_data - flat_data.mean()
    scaled_data = scaled_data * 10000  # do it like FSL
    # reshape to volume * TRs
    scaled_data = np.reshape(scaled_data, img_data.shape)

    # create Nifti image
    scaled_img = nib.Nifti1Image(scaled_data,
                                 img.affine,
                                 header=img.header)

    # save it as file
    run = re.search(r'run-\d{1}', in_fpath).group()
    out_fpath = os.path.join(subj,
                             f'{subj}_task-{stimulus}_{run}_bold_filtered' +
                             '.nii.gz')
    nib.save(scaled_img, out_fpath)

    return None


if __name__ == "__main__":
    # find filtered_func_data.nii.gz for all subjects/runs
    vis_filtered_fpathes = find_files(VIS_FILTERED_PATTERN)
    ao_filtered_fpathes = find_files(AO_FILTERED_PATTERN)
    av_filtered_fpathes = find_files(AV_FILTERED_PATTERN)

    # slice the subject string from the file pathes
    subjs_pathes = ao_filtered_fpathes
    subjs = [re.search(r'sub-..', string).group() for string in subjs_pathes]
    # some filtering (which is probably not necessary)
    subjs = sorted(list(set(subjs)))

    for subj in subjs:
        print('\nProcessing', subj)
        # created subject-specific directories in case they do not exist
        os.makedirs(os.path.join(subj), exist_ok=True)

        # filter file pathes for current subject
        subj_vis_fpathes = [fpath for fpath in vis_filtered_fpathes
                            if subj in fpath]
        subj_ao_fpathes = [fpath for fpath in ao_filtered_fpathes
                           if subj in fpath]
        subj_av_fpathes = [fpath for fpath in av_filtered_fpathes
                           if subj in fpath]

        # loop through the 4d data of the visual localizer
        for vis_img_fpath in subj_vis_fpathes:
            # normalize the data
            grand_mean_for_4d_file(vis_img_fpath, 'visloc')

        # loop through the 4d data of the audio-description
        for ao_img_fpath in subj_ao_fpathes:
            # normalize the data
            grand_mean_for_4d_file(ao_img_fpath, 'aomovie')

        # loop through the 4d data of the movie
        for av_img_fpath in subj_av_fpathes:
            # normalize the data
            grand_mean_for_4d_file(av_img_fpath, 'avmovie')

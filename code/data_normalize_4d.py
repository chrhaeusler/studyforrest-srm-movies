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


def normalize_4d_file(in_fpath, stimulus='aomovie'):
    '''
    '''
    print("loading", in_fpath)

    if not os.path.exists(in_fpath):
        subprocess.call(['datalad', 'get', in_fpath])

    img = nib.load(in_fpath)
    img_data = img.get_fdata()
    print('Data shape - before reshaping: ', img_data.shape)

    # flatten the data
    flat_data = np.reshape(
        img_data,
        (
            img_data.shape[0] * img_data.shape[1] * img_data.shape[2],
            img_data.shape[3]
            )
    )

    flat_data = np.transpose(flat_data)
    print('Data shape - after reshaping: ', flat_data.shape)

    # normalize the data (Z-scoring)
    # grand mean scaling for 4D data:
    # An analysis step in which the voxel values in every
    # image are divided by the average global mean intensity of the whole
    # session. This effectively removes any mean global differences in
    # intensity between sessions.
    scaler = preprocessing.StandardScaler().fit(flat_data)
    norm_flat_data = scaler.transform(flat_data)

    # do some checks
    voxel_mean = np.mean(norm_flat_data, axis=0)
    voxel_std = np.std(norm_flat_data, axis=0)
    print('Number of voxels is %d' % len(voxel_mean))
    print('Mean of first few voxels: ', voxel_mean[0:10])
    print('Std.Dev. of first few voxels: ', voxel_std[0:10])

    # get the data back into original 4D shape
    normalized_data = np.transpose(norm_flat_data)
    normalized_data = np.reshape(normalized_data, img_data.shape)
    normalized_img = nib.Nifti1Image(normalized_data,
                                     img.affine,
                                     header=img.header)

    # save it as file
    run = re.search(r'run-\d{1}', in_fpath).group()
    out_fpath = os.path.join(subj,
                             f'{subj}_task-{stimulus}_{run}_bold_filtered' +
                             '.nii.gz')
    nib.save(normalized_img, out_fpath)

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
            normalize_4d_file(vis_img_fpath, 'visloc')

        # loop through the 4d data of the audio-description
        for ao_img_fpath in subj_ao_fpathes:
            # normalize the data
            normalize_4d_file(ao_img_fpath, 'aomovie')

        # loop through the 4d data of the movie
        for av_img_fpath in subj_av_fpathes:
            # normalize the data
            normalize_4d_file(av_img_fpath, 'avmovie')

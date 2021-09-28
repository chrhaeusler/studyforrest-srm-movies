#!/usr/bin/env python3
'''
created on Mon March 29th 2021
author: Christian Olaf Haeusler
'''

from glob import glob
from nilearn.input_data import NiftiMasker
from sklearn import preprocessing
import argparse
import nibabel as nib
import numpy as np
import os
import re


# constants
GRP_PPA_PTTRN = 'sub-??/masks/in_bold3Tp2/grp_PPA_bin.nii.gz'
AO_FOV_MASK = 'sub-??/masks/in_bold3Tp2/audio_fov.nii.gz'
GM_MASK = 'sub-??/masks/in_bold3Tp2/gm_bin_dil_fov.nii.gz'

AOAV_FILE_PTTRN = 'sub-??/sub-??_task-a?movie_run-?_bold_filtered.nii.gz'
VIS_FILE_PTTRN = 'sub-??/sub-??_task-visloc_run-?_bold_filtered.nii.gz'


def parse_arguments():
    '''
    '''
    parser = argparse.ArgumentParser(
        description='masks and concatenates runs of AO & AV stimulus'
    )

    parser.add_argument('-sub',
                        required=False,
                        default='sub-01',
                        help='subject to process (e.g. "sub-01")')

    parser.add_argument('-outdir',
                        required=False,
                        default='test',
                        help='output directory (e.g. "sub-01")')

    args = parser.parse_args()

    sub = args.sub
    outdir = args.outdir

    return sub, outdir


def find_files(pattern):
    '''
    '''
    found_files = glob(pattern)
    found_files = sort_nicely(found_files)

    return found_files


def sort_nicely(l):
    '''Sorts a given list in the way that humans expect
    '''
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    l.sort(key=alphanum_key)

    return l


def load_mask(subj):
    '''
    '''
    # open the mask of cortices (at the moment it justs the union of
    # individual PPAs)
    grp_ppa_fpath = GRP_PPA_PTTRN.replace('sub-??', subj)
    print('PPA GRP mask:\t', grp_ppa_fpath)

    # subject-specific field of view in audio-description
    ao_fov_mask = AO_FOV_MASK.replace('sub-??', subj)
    print('ind. FoV mask:\t', ao_fov_mask)

    # load the masks
    grp_ppa_img = nib.load(grp_ppa_fpath)
    ao_fov_img = nib.load(ao_fov_mask)

    # (dilated) gray matter mask; see constat at script's top
    #gm_fpath = GM_MASK.replace('sub-??', subj)
    #gm_img = nib.load(gm_fpath)
    #final_mask_data = grp_ppa_img.get_fdata() * gm_img.get_fdata()

    final_mask_data = grp_ppa_img.get_fdata() * ao_fov_img.get_fdata()
    final_mask_img = nib.Nifti1Image(final_mask_data,
                                     grp_ppa_img.affine,
                                     header=grp_ppa_img.header)

    return final_mask_img


def process_infiles(in_fpathes):
    '''
    '''
    # load first run
    print(in_fpathes[0])
    first_img = nib.load(in_fpathes[0])
    # mask the 4D image and get the data as np.ndarray
    masked_data = nifti_masker.fit_transform(first_img)

    # perform z-scoring of the run
    scaler = preprocessing.StandardScaler()
    scaler.fit(masked_data)
    masked_data = scaler.transform(masked_data)
    # reshape to voxel x TRs
    masked_data = np.transpose(masked_data)
    print('current run:\t', masked_data.shape)

    # loop over the rest of runs
    for run, in_fpath in enumerate(in_fpathes[1:]):  # first index is 'first_img'
        run = run + 2
        # load current loop's run
        print(in_fpath)
        new_img = nib.load(in_fpath)
        # mask the 4D image and get the data as np.ndarray
        masked_new_data = nifti_masker.fit_transform(new_img)

        # perform z-scoring of the run
        scaler = preprocessing.StandardScaler()
        scaler.fit(masked_new_data)
        masked_new_data = scaler.transform(masked_new_data)
        # reshape to voxel x TRs
        masked_new_data = np.transpose(masked_new_data)

        if 'aomovie_run-8_bold_filtered' in in_fpath and 'sub-04' not in in_fpath:
            print('slicing run-8')
            masked_new_data = masked_new_data[:, :263]

        # concatenate current time-series to previous time-series
        masked_data = np.concatenate(
            (masked_data, masked_new_data), axis=1
        )

        # print current size of current run
        print('current run:\t', masked_new_data.shape)
        # print current size of the now extended time-series
        print('processed runs:\t', masked_data.shape)

    return masked_data


if __name__ == "__main__":
    # read command line arguments
    subj, out_dir = parse_arguments()
    subj = subj.strip('/')

    print('\nProcessing', subj)

    # load the mask
    mask_img = load_mask(subj)
    # create instance of NiftiMasker used to mask the 4D time-series
    # which will be loaded next
    nifti_masker = NiftiMasker(mask_img=mask_img)

    # AUDIO-DESCRIPTIO & MOVIE
    print('\nProcessing data of audio-description & movie')

    # get the files for current subject
    aoav_pattern = AOAV_FILE_PTTRN.replace('sub-??', subj)
    aoav_fpathes = find_files(aoav_pattern)

    # process the files
    masked_data = process_infiles(aoav_fpathes)

    # prepare to save
    out_file = f'{subj}_task_aomovie-avmovie_run-1-8_bold-filtered.npy'
    out_fpath = os.path.join(out_dir, out_file)
    # only needed is path "test" is used
    os.makedirs(os.path.dirname(out_fpath), exist_ok=True)
    # save it
    np.save(out_fpath, masked_data)

    # VISUAL LOCALIZER
    print('\nProcessing data of visual localizer')

    # get the files for current subject
    vis_pattern = VIS_FILE_PTTRN.replace('sub-??', subj)
    vis_fpathes = find_files(vis_pattern)

    # process the files
    masked_data = process_infiles(vis_fpathes)

    # prepare to save
    out_file = f'{subj}_task_visloc_run-1-4_bold-filtered.npy'
    out_fpath = os.path.join(out_dir, out_file)
    # only needed if path "test" is used
    os.makedirs(os.path.dirname(out_fpath), exist_ok=True)
    # save it
    np.save(out_fpath, masked_data)

#!/usr/bin/env python3
'''
author: Christian Olaf Häusler
created on Wednesday, 31 March 2021
'''

from glob import glob
import subprocess
import nibabel as nib
import numpy as np
import os
import re


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
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    l.sort(key=alphanum_key)

    return l


def mni_2_ind_bold3Tp2(mni_mask, out_fpath, ind_ref, ind_warp, premat):
    '''
    '''
    if not os.path.exists(ind_ref):
        subprocess.call(['datalad', 'get', ind_ref])

    if not os.path.exists(ind_warp):
        subprocess.call(['datalad', 'get', ind_warp])

    if not os.path.exists(premat):
        subprocess.call(['datalad', 'get', premat])

    subprocess.call(
        ['applywarp',
         '-i', mni_mask,
         '-o', out_fpath,
         '-r', ind_ref,
         '-w', ind_warp,
         '--premat=' + premat
         ]
    )

    return None


# main program #
if __name__ == "__main__":
    # some hardcoded sources
    # path of the subdataset providing templates and transformatiom matrices
    TNT_DIR = 'inputs/studyforrest-data-templatetransforms'
    # filename for pre-transform (affine matrix)
    XFM_MAT = os.path.join(
        TNT_DIR,
        'templates/grpbold3Tp2/xfm/',
        'mni2tmpl_12dof.mat'
    )

    # the path to check for which subjects we have (filtered) functional data
    # that were used to localize the PPA via movie and audio-description
    SUBJS_PATH_PATTERN = 'inputs/studyforrest-ppa-analysis/sub-??'
    # the path that contains mask (input and output purpose)
    ROIS_PATH = 'rois-and-masks'

    subjs_pathes = find_files(SUBJS_PATH_PATTERN)
    subjs = [re.search(r'sub-..', string).group() for string in subjs_pathes]
    # some filtering (which is probably not necessary)
    subjs = sorted(list(set(subjs)))

    masks_in_mni = find_files(os.path.join(ROIS_PATH, 'in_mni', '*.*'))

    for subj in subjs:
        # create the subject-specific folder in case it does not exist
        os.makedirs(os.path.join(ROIS_PATH, subj), exist_ok=True)

        for mask_in_mni in masks_in_mni:
            # change the output path to the current subject
            out_fpath = mask_in_mni.replace('in_mni', subj)
            # the path of the (individual) reference image
            subj_ref = os.path.join(TNT_DIR, subj, 'bold3Tp2/brain.nii.gz')
            # the volume providing warp/coefficient
            subj_warp = os.path.join(
                TNT_DIR,
                subj,
                'bold3Tp2/in_grpbold3Tp2/'
                'tmpl2subj_warp.nii.gz'
            )
            # the (affine) pre-transformation matrix
            premat = XFM_MAT

            # warp mask (e.g. Havard-Oxford & MNI probabilistic)
            # from mni into individual subject spaces
            mni_2_ind_bold3Tp2(
                mask_in_mni,
                out_fpath,
                subj_ref,
                subj_warp,
                premat
            )

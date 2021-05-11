#!/usr/bin/env python3
'''
author: Christian Olaf Häusler
created on Monday, 12 April 2021
'''

from glob import glob
import subprocess
import os
import re


def find_files(pattern):
    '''
    '''
    def sort_nicely(l):
        '''Sorts a given list in the way that humans expect
        '''
        convert = lambda text: int(text) if text.isdigit() else text
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        l.sort(key=alphanum_key)

        return l

    found_files = glob(pattern)
    found_files = sort_nicely(found_files)

    return found_files


def tw1_2_bold3Tp2(t1w_mask, out_fpath, ind_ref, xmf_matrix):
    '''
    '''
    if not os.path.exists(t1w_mask):
        subprocess.call(['datalad', 'get', t1w_mask])

    if not os.path.exists(ind_ref):
        subprocess.call(['datalad', 'get', ind_ref])

    if not os.path.exists(xmf_matrix):
        subprocess.call(['datalad', 'get', xmf_matrix])

    subprocess.call([
        'flirt',
        '-in', t1w_mask,
        '-out', out_fpath,
        '-ref', ind_ref,
        '-init', xmf_matrix,
        '-applyxfm'
        ]
    )

    return None


# main program #
if __name__ == "__main__":
    # some hardcoded sources
    # path of the subdataset providing templates and transformatiom matrices
    TNT_DIR = 'inputs/studyforrest-data-templatetransforms'
    # pattern of relevant t1w masks
    T1W_MASK_PATTERN = 'brain_seg*.nii.gz'
    # the path to check for which subjects we have (filtered) functional data
    # that were used to localize the PPA via movie and audio-description
    SUBJS_PATH_PATTERN = 'inputs/studyforrest-ppa-analysis/sub-??'
    # the path that contains mask (input and output purpose)
    ROIS_PATH = 'masks'

    # find all subjects for which we have filtered 4D data
    subjs_pathes = find_files(SUBJS_PATH_PATTERN)
    # get the subjects' strings (e.g. "subj-01")
    subjs = [re.search(r'sub-..', string).group() for string in subjs_pathes]
    # some filtering
    subjs = sorted(list(set(subjs)))

    # loop through the subjects
    for subj in subjs:
        # create the subject-specific folder in case it does not exist
        out_path = os.path.join(subj, ROIS_PATH, 'in_bold3Tp2')
        os.makedirs(out_path, exist_ok=True)

        # find all mask that need to be aligned to bold3Tp2
        t1w_path = os.path.join(TNT_DIR, subj, 't1w')
        t1w_masks = find_files(os.path.join(t1w_path, T1W_MASK_PATTERN))

        # individualize inputs for flirt
        ref_brain = os.path.join(t1w_path, 'in_bold3Tp2/brain.nii.gz')
        matrix = os.path.join(t1w_path, 'in_bold3Tp2/xfm_6dof.mat')

        # loop trough the masks
        for t1w_mask in t1w_masks:
            # create output path filename
            mask_fname = os.path.basename(t1w_mask)
            out_fpath = os.path.join(out_path, mask_fname)

            # call the function that calls FSL's flirt
            tw1_2_bold3Tp2(t1w_mask, out_fpath, ref_brain, matrix)

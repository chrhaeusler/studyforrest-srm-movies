#!/usr/bin/env python3
'''
created on Mon March 29th 2021
author: Christian Olaf Haeusler
'''

from glob import glob
from scipy import stats
# from nilearn.input_data import NiftiMasker, MultiNiftiMasker
import brainiak.funcalign.srm
import argparse
# import nibabel as nib
import ipdb
import numpy as np
import os
import random
import re


# constants
AOAV_TRAIN_PATTERN = 'sub-??/sub-??_ao-av-vis_concatenated_zscored.npy'
VIS_TRAIN_PATTERN = 'sub-??/sub-??_task_visloc_run-1-4_bold-filtered.npy'


def parse_arguments():
    '''
    '''
    parser = argparse.ArgumentParser(
        description='fits SRM to data from all subjects except given subject'
    )

    parser.add_argument('-sub',
                        required=False,
                        default='sub-01',
                        help='subject to leave out (e.g. "subj-01")')

    parser.add_argument('-outdir',
                        required=False,
                        default='test',
                        help='output directory (e.g. "sub-01")')

    parser.add_argument('-nfeat',
                        required=False,
                        default='50',
                        help='number of features (shared responses)')

    parser.add_argument('-niter',
                        required=False,
                        default='20',
                        help='number of iterations')

    args = parser.parse_args()

    sub = args.sub
    outdir = args.outdir
    n_feat = int(args.nfeat)
    n_iter = int(args.niter)

    return sub, outdir, n_feat, n_iter


def find_files(pattern):
    '''
    '''
    def sort_nicely(l):
        '''Sorts a given list in the way that humans expect
        '''
        convert = lambda text: int(text) if text.isdigit() else text
        alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
        l.sort(key=alphanum_key)

        return l

    found_files = glob(pattern)
    found_files = sort_nicely(found_files)

    return found_files


def load_and_zscore(aoav_fpath, vis_fpath):
    '''
    '''
    # load data of audio-description (lasrt 75TRs are already cutted)
    # & visual localizer
    aoav_array = np.load(aoav_fpath)
    # load data of visual localizer
    vis_array = np.load(vis_fpath)

    # concat AOAV data and VIS data
    aoavvis_array = np.concatenate([aoav_array, vis_array],
                                   axis=1)

    # perform zscoring across concatenated experiments
    zscored_aoavvis_array = stats.zscore(aoavvis_array,
                                         axis=1,
                                         ddof=1)

    return zscored_aoavvis_array


def fit_srm(list_of_arrays, out_dir):
    '''Fits the SRM and saves it to files
    '''
    srm = brainiak.funcalign.srm.SRM(features=n_feat, n_iter=n_iter)

    # fit the SRM model
    print(f'fitting SRM to data of all subjects except {subj}...')
    srm.fit(list_of_arrays)

    return srm


if __name__ == "__main__":
    # read command line arguments
    subj, out_dir, n_feat, n_iter = parse_arguments()

    # find all input files
    aoavvis_fpathes = find_files(AOAV_TRAIN_PATTERN)
    # filter for non-current subject
    aoavvis_fpathes = [fpath for fpath in aoavvis_fpathes if subj not in fpath]

    aoavvis_arrays = []
    # loops through subjects (one concatenated / masked time-series per sub)
    for aoavvis_fpath in aoavvis_fpathes:
        # load the data
        print(f'loading {aoavvis_fpath}')
        aoavvis_array = np.load(aoavvis_fpath)
        # append to the list of arrays (containing all subjects' arrays)
        aoavvis_arrays.append(aoavvis_array)

    # fit the SRM model
    srm_aoavvis = fit_srm(aoavvis_arrays, out_dir)

    # prepare saving results as pickle
    out_file = f'srm-ao-av-vis_feat{n_feat}-iter{n_iter}.npz'
    out_fpath = os.path.join(out_dir, subj, 'models', out_file)
    # create (sub)directories
    os.makedirs(os.path.dirname(out_fpath), exist_ok=True)
    # save it
    srm_aoavvis.save(out_fpath)
    print('SRM saved to', out_fpath)

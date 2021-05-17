#!/usr/bin/env python3
'''
created on Mon May 17th 2021
author: Christian Olaf Haeusler
'''

from glob import glob
# from nilearn.input_data import NiftiMasker, MultiNiftiMasker
import matplotlib.pyplot as plt
import brainiak.funcalign.srm
import argparse
# import nibabel as nib
import numpy as np
import os
import re


def parse_arguments():
    '''
    '''
    parser = argparse.ArgumentParser(
        description='load a pickled SRM and do some plotting'
    )

    parser.add_argument('-sub',
                        required=False,
                        default='sub-02',
                        help='subject to leave out (e.g. "subj-01")')

    parser.add_argument('-indir',
                        required=False,
                        default='test',
                        help='output directory (e.g. "sub-01")')

    parser.add_argument('-nfeat',
                        required=False,
                        default='10',
                        help='number of features (shared responses)')

    parser.add_argument('-niter',
                        required=False,
                        default='20',
                        help='number of iterations')

    args = parser.parse_args()

    sub = args.sub
    indir = args.indir
    n_feat = int(args.nfeat)
    n_iter = int(args.niter)

    return sub, indir, n_feat, n_iter


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


if __name__ == "__main__":
    # read command line arguments
    subj, in_dir, n_feat, n_iter = parse_arguments()
    # save model as (zipped) pickle variable
    in_fpath = os.path.join(
        in_dir, f'{subj}_srm_feat{n_feat}-iter{n_iter}.npz'
    )

    # make np.load work with allow_pickle=True
    # save np.load
    np_load_old = np.load
    # modify the default parameters of np.load
    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
    # np.load = lambda *a: np_load_old(*a, allow_pickle=True)
    # load the pickle file
    srm = brainiak.funcalign.srm.load(in_fpath)
    # change np.load() back to normal
    np.load = np_load_old

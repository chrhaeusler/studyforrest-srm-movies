#!/usr/bin/env python3
'''
created on Mon March 29th 2021
author: Christian Olaf Haeusler
'''

from glob import glob
# from nilearn.input_data import NiftiMasker, MultiNiftiMasker
import brainiak.funcalign.srm
import argparse
# import nibabel as nib
import copy
import ipdb
import numpy as np
import os
import random
import re


# constants
IN_PATTERN_NAT = 'sub-??/sub-??_task_aomovie-avmovie_run-1-8_bold-filtered.npy'
IN_PATTERN_VIS = 'sub-??/sub-??_task_visloc_run-1-4_bold-filtered.npy'

# and vary the amount of TRs used for alignment
STARTS_ENDS = [
    (0, 451),      # AO, 1 run
    (0, 892),      # AO, 2 runs
    (0, 1330),     # AO, 3 runs
    (0, 1818),     # AO, 4 runs
    (0, 2280),     # AO, 5 runs
    (0, 2719),     # AO, 6 runs
    (0, 3261),     # AO, 7 runs
    (0, 3524),     # AO, 8 runs
    (3524, 3975),  # AV, 1 run
    (3524, 4416),  # AV, 2 runs
    (3524, 4854),  # AV, 3 runs
    (3524, 5342),  # AV, 4 runs
    (3524, 5804),  # AV, 5 runs
    (3524, 6243),  # AV, 6 runs
    (3524, 6785),  # AV, 7 runs
    (3524, 7123),  # AV, 8 runs
    # (0, 7123),      # AO & AV
    (7123, 7123 + 1 * 156),  # VIS, 1 run
    (7123, 7123 + 2 * 156),  # VIS, 2 run
    (7123, 7123 + 3 * 156),  # VIS, 3 run
    (7123, 7123 + 4 * 156)  # VIS, 4 run
]

def parse_arguments():
    '''
    '''
    parser = argparse.ArgumentParser(
        description='load a pickled SRM and do some plotting'
    )


    parser.add_argument('-indir',
                        required=False,
                        default='test',
                        help='input directory')

    parser.add_argument('-sub',
                        required=False,
                        default='sub-01',
                        help='subject to leave out (e.g. "subj-01")')

    parser.add_argument('-model',
                        required=False,
                        default='srm-ao-av-vis',
                        help='the model file\'s base name')

    parser.add_argument('-nfeat',
                        required=False,
                        default='10',
                        help='number of features (shared responses)')

    parser.add_argument('-niter',
                        required=False,
                        default='30',
                        help='number of iterations')

    args = parser.parse_args()

    indir = args.indir
    sub = args.sub
    model = args.model
    n_feat = int(args.nfeat)
    n_iter = int(args.niter)

    return indir, sub, model, n_feat, n_iter


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


def load_srm(in_fpath):
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

    return srm


if __name__ == "__main__":
    # read command line arguments
    in_dir, sub, model, n_feat, n_iter = parse_arguments()

    for modelStart, modelEnd in STARTS_ENDS:
        print(f'\nUsing {model}, {modelStart}-{modelEnd}')

        in_fpath = os.path.join(
            in_dir,
            sub,
            'models',  # hard coded FTW
            f'{model}_feat{n_feat}-iter{n_iter}.npz'
        )

        # load the srm from file
        print('Loading SRM:', in_fpath)
        srm = load_srm(in_fpath)

        # leave the original srm untouched but copy it
        srm_sliced = copy.copy(srm)
        srm_sliced.s_ = srm_sliced.s_[:, modelStart:modelEnd]

        # load the time series of the paradigms
        # AO and AV are concatenated
        # VIS is separate file
        # adjust the indices accordingly
        if modelStart < 7123:
            in_fpath = IN_PATTERN_NAT.replace('sub-??', sub)
            paradigmStart, paradigmEnd = modelStart, modelEnd
        elif modelStart == 7123:
            in_fpath = IN_PATTERN_VIS.replace('sub-??', sub)
            paradigmStart, paradigmEnd = modelStart - 7123, modelEnd - 7123

        print('Loading data:', in_fpath)

        array = np.load(in_fpath)
        array_sliced = array[:, paradigmStart:paradigmEnd]
        # print(array_sliced.shape)
        w_matrix = srm_sliced.transform_subject(array_sliced)

        # save the matrix
        # create name of output file
        out_file = f'wmatrix_{model}_feat{n_feat}_{modelStart}-{modelEnd}.npy'

        # just take the input math as the output path
        out_fpath = os.path.join(
            in_dir,
            sub,
            'matrices',  # hard coded FTW
            out_file
        )

        # create (sub)directories
        os.makedirs(os.path.dirname(out_fpath), exist_ok=True)
        # save it
        np.save(out_fpath, w_matrix)
        print(f'weight matrix for {sub} saved to', out_fpath)

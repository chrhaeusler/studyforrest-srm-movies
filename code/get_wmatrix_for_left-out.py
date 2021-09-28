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
IN_PATTERN = 'sub-??/sub-??_task_aomovie-avmovie_run-1-8_bold-filtered.npy'

# which TRs do we wanna use?
# AO indice: 0 to 3598
# AV indices: 3599 to 7122
# the last 75 TRs of AV were cutted because they are missing in sub-04
# start = 0  # 3599
# end = 451  # 3599 + 451 + 441 + 438 + 488 + 462 + 439 + 542 + (338-75)


def parse_arguments():
    '''
    '''
    parser = argparse.ArgumentParser(
        description='load a pickled SRM and do some plotting'
    )


    parser.add_argument('-indir',
                        required=False,
                        default='test',
                        help='output directory (e.g. "sub-01")')

    parser.add_argument('-model',
                        required=False,
                        default='srm-ao-av-vis',
                        help='the model (e.g. "srm")')

    parser.add_argument('-nfeat',
                        required=False,
                        default='30',
                        help='number of features (shared responses)')

    parser.add_argument('-niter',
                        required=False,
                        default='30',
                        help='number of iterations')

    args = parser.parse_args()

    indir = args.indir
    model = args.model
    n_feat = int(args.nfeat)
    n_iter = int(args.niter)

    return indir, model, n_feat, n_iter


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
    in_dir, model, n_feat, n_iter = parse_arguments()

    SUBJS_PATH_PATTERN = 'sub-??'
    subjs_pathes = find_files(SUBJS_PATH_PATTERN)
    subjs = [re.search(r'sub-..', string).group() for string in subjs_pathes]
    # some filtering
    subjs = sorted(list(set(subjs)))

    # loob through the models
    models = [
        'srm-ao-av',
        'srm-ao-av-vis'
    ]
    # and vary the amount of TRs used for alignment
    starts_ends = [
        (0, 451),
        (0, 3524),
        (3524, 3975),
        (3524, 7123),
        (0, 7123)
    ]

    for model in models:
        for start, end in starts_ends:
            print(f'using {model}, {start}-{end}')
            for subj in subjs:
                in_fpath = os.path.join(
                    in_dir, subj, f'{model}_feat{n_feat}-iter{n_iter}.npz'
                )

                # load the srm from file
                print('Loading', in_fpath)
                srm = load_srm(in_fpath)

                # leave the original srm untouched but copy it
                srm_sliced = copy.copy(srm)
                srm_sliced.s_ = srm_sliced.s_[:, start:end]

                in_fpath = IN_PATTERN.replace('sub-??', subj)
                array = np.load(in_fpath)
                array = array[:, start:end]
                w_matrix = srm_sliced.transform_subject(array)

                # save the matrix
                out_file = f'wmatrix_{model}_feat{n_feat}_{start}-{end}.npy'
                out_fpath = os.path.join(in_dir, subj, out_file)
                # create (sub)directories
                os.makedirs(os.path.dirname(out_fpath), exist_ok=True)
                # save it
                np.save(out_fpath, w_matrix)
                print(f'weight matrix for {subj} saved to', out_fpath, '\n')

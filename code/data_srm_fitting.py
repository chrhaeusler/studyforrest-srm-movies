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
import numpy as np
import os
import re


# constants
IN_PATTERN = 'sub-??/sub-??_task_aomovie-avmovie_run-1-8_bold-filtered.npy'


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


if __name__ == "__main__":
    # read command line arguments
    subj, out_dir, n_feat, n_iter = parse_arguments()

    # find all input files
    in_fpathes = find_files(IN_PATTERN)
    # filter for non-current subject
    in_fpathes = [fpath for fpath in in_fpathes if subj not in fpath]

    movie_arrays = []
    for in_fpath in in_fpathes:
        array = np.load(in_fpath)
        dim = array.shape

        # check (hard coded) expected number of TRs
        if dim[1] == 7198:
            movie_arrays.append(array)
            print(in_fpath[:6], dim)
        else:
            # do some zero padding for sub-04 who has ~ 75 TRs missing in
            # run 8
            zero_padded = np.zeros([array.shape[0], 7198], dtype=np.float32)
            print(in_fpath[:6], dim, '(before)')
            zero_padded[:array.shape[0], :array.shape[1]] = array
            print(in_fpath[:6], zero_padded.shape, '(after)')
            movie_arrays.append(zero_padded)

    # concat array; but: brainaik expects list of arrays
    # movie_data = np.concatenate(arrays, axis=0)

    # Create the SRM object
    srm = brainiak.funcalign.srm.SRM(features=n_feat, n_iter=n_iter)

    # Fit the SRM data
    # fit the model
    print(f'Fitting SRM to data of all subjects except {subj}...')
    # TESTING PURPOSE: take only a slice of all subjects data
    # movie_arrays = [array[:, :150] for array in movie_arrays]
    # actuall model fitting
    srm.fit(movie_arrays)

    # prepare saving results as pickle
    out_file = f'{subj}_srm_feat{n_feat}-iter{n_iter}.npz'
    out_fpath = os.path.join(out_dir, out_file)
    # create (sub)directories
    os.makedirs(os.path.dirname(out_fpath), exist_ok=True)
    # save it
    srm.save(out_fpath)
    print('SRM saved to', out_fpath)

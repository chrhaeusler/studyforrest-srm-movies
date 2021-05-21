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


def zero_padding(in_fpath):
    '''performs zero padding of an array in case no. of columns is != 7198
    '''
    # check (hard coded) expected number of TRs
    array = np.load(in_fpath)
    dim = array.shape

    if dim[1] == 7198:
        print(in_fpath[:6], dim)
        return_array = array

    else:
        # do some zero padding for sub-04 who has ~ 75 TRs missing in
        # run 8
        return_array = np.zeros([array.shape[0], 7198], dtype=np.float32)
        print(in_fpath[:6], dim, '(before)')
        return_array[:array.shape[0], :array.shape[1]] = array
        print(in_fpath[:6], return_array.shape, '(after)')

    return return_array


def array_cutting(in_fpath):
    '''performs zero padding of an array in case no. of columns is != 7198
    '''
    # check (hard coded) expected number of TRs
    array = np.load(in_fpath)
    dim = array.shape

    if dim[1] > 7123:
        # slice the array to length of 7123 TRs
        return_array = array[:, :7123]
        new_dim = return_array.shape
        print(in_fpath[:6], new_dim, '(after cutting)')
    elif dim[1] == 7123:
        # correct length -> do noting
        return_array = array
        print(in_fpath[:6], dim, '(unchanged)')
    else:
        raise ValueError('unexpected number of TRs')

    return return_array


def fit_srm_and_save(list_of_arrays, out_dir):
    '''Fits the SRM and saves it to files

    To Do: change from global to local variables
    '''
    srm = brainiak.funcalign.srm.SRM(features=n_feat, n_iter=n_iter)

    # Fit the SRM data
    # fit the model
    print(f'Fitting SRM to data of all subjects except {subj}...')
    # TESTING PURPOSE: take only a slice of all subjects data
    # movie_arrays = [array[:, :150] for array in movie_arrays]
    # actuall model fitting
    srm.fit(list_of_arrays)

    # prepare saving results as pickle
    out_file = f'{subj}_srm_feat{n_feat}-iter{n_iter}.npz'
    out_fpath = os.path.join(out_dir, out_file)
    # create (sub)directories
    os.makedirs(os.path.dirname(out_fpath), exist_ok=True)
    # save it
    srm.save(out_fpath)
    print('SRM saved to', out_fpath)

    return None


if __name__ == "__main__":
    # read command line arguments
    subj, out_dir, n_feat, n_iter = parse_arguments()

    # find all input files
    in_fpathes = find_files(IN_PATTERN)
    # filter for non-current subject
    in_fpathes = [fpath for fpath in in_fpathes if subj not in fpath]

    movie_arrays = []
    for in_fpath in in_fpathes:
        # ~75 TRs of run-8 in sub-04 are missing
        # Do:
        # a) perform zero padding
        # corrected_array = zero_padding(in_fpath)
        # b) cutting of all other arrays
        corrected_array = array_cutting(in_fpath)
        # populate the list of arrays (as expected by brainiac)
        movie_arrays.append(corrected_array)

    # Create the SRM object
    # fit_srm_and_save(movie_arrays, out_dir)

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
import ipdb
import numpy as np
import os
import random
import re


# constants
TRAIN_PATTERN = 'sub-??/sub-??_task_aomovie-avmovie_run-1-8_bold-filtered.npy'


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

    if dim[1] > 7123:  # all except sub-04
        print(in_fpath[:6], dim, '(before cutting)')
        # in case AO data come before the AV data
        # cut the last 75 TRs from the audio-description's data
        ao = array[:, :3599-75]
        # take all of the movie's data
        av = array[:, 3599:]
        return_array = np.concatenate([ao, av], axis=1)

#         # in case the ao data follow the av dats
#         # slice the array to length of 7123 TRs
#         return_array = array[:, :7123]

        new_dim = return_array.shape
        print(in_fpath[:6], new_dim, '(after cutting)')

    elif dim[1] == 7123:
        # correct length -> do noting
        return_array = array
        print(in_fpath[:6], dim, '(unchanged)')
    else:
        raise ValueError('unexpected number of TRs')

    return return_array


def fit_srm(list_of_arrays, out_dir):
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



    return srm


def shuffle_all_arrays(all_arrays):
    '''
    '''
    timings = [0, 451, 441, 438, 488, 462, 439, 542, 338-75]
    timings = timings + timings[1:-1] + [338]  # add AO to AO
    starts = [sum(timings[0:idx+1]) for idx, value in enumerate(timings)]
    starts_ends = [[x, y] for x, y in zip(starts[:-1], starts[1:])]
    # substitute the last index for '-1'
    starts_ends[-1][1] = ''

    shuffled_subjs = []
    for subject, array in enumerate(all_arrays):
        random.shuffle(starts_ends)

        shuffled_blocks_arrays = []
        for start, end in starts_ends:
            # print(start, end)
            # append the current block
            if end:
                shuffled_blocks_arrays.append(array[:, start:end])
            else:
                shuffled_blocks_arrays.append(array[:, start:])

        # manipulate the array
        shuffled_blocks = np.concatenate(shuffled_blocks_arrays, axis=1)
        # concatenate the blocks
        shuffled_subjs.append(shuffled_blocks)

    return shuffled_subjs


if __name__ == "__main__":
    # read command line arguments
    subj, out_dir, n_feat, n_iter = parse_arguments()

    # find all input files
    train_fpathes = find_files(TRAIN_PATTERN)
    # filter for non-current subject
    train_fpathes = [fpath for fpath in train_fpathes if subj not in fpath]

    all_arrays = []
    # loops through subjects (one concatenated / masked time-series per sub)
    for train_fpath in train_fpathes:
        # ~75 TRs of run-8 in sub-04 are missing
        # Do:
        # a) perform zero padding
        # corrected_array = zero_padding(in_fpath)
        # b) cutting of all other arrays
        corrected_array = array_cutting(train_fpath)
        # populate the list of arrays (as expected by brainiac)
        all_arrays.append(corrected_array)

        # shuffle the current subject's array

    # Create the SRM object
    srm = fit_srm(all_arrays, out_dir)

    # prepare saving results as pickle
    model = 'srm-ao-av'
    out_file = f'{subj}_{model}_feat{n_feat}-iter{n_iter}.npz'
    out_fpath = os.path.join(out_dir, out_file)
    # create (sub)directories
    os.makedirs(os.path.dirname(out_fpath), exist_ok=True)
    # save it
    srm.save(out_fpath)
    print('SRM saved to', out_fpath)

    # negative control:
    # shuffle the arrays before fitting the model
    # always take the same seed for every subject
    # by deriving it from the subject's number
    random.seed(int(subj[-2:]))
    shuffled_arrays = shuffle_all_arrays(all_arrays)
    srm = fit_srm(shuffled_arrays, out_dir)

    # prepare saving results as pickle
    out_file = f'{subj}_{model}_feat{n_feat}-iter{n_iter}_shuffled.npz'
    out_fpath = os.path.join(out_dir, out_file)
    # create (sub)directories
    os.makedirs(os.path.dirname(out_fpath), exist_ok=True)
    # save it
    srm.save(out_fpath)
    print('SRM saved to', out_fpath)

    #### EXTEND THE ARRAY PER SUBJECT WITH THE DATA OF THE VIS EXPERIMENT HERE

    ### FIT THE MODEL

    ### SAVE THE MODEL

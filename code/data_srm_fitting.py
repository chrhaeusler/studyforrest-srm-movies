#!/usr/bin/env python3
'''
created on Mon March 29th 2021
author: Christian Olaf Haeusler

todo:
    needs factorization obviously
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
AOAV_TRAIN_PATTERN = 'sub-??/sub-??_task_aomovie-avmovie_run-1-8_bold-filtered.npy'
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
    '''cuts the last 75 TRs from the audio-description's 8th run in order to
    have equal length across subjects (last 75 TRs are missing in sub-04)
    '''
    # check (hard coded) expected number of TRs
    array = np.load(in_fpath)
    print('\nloading', in_fpath)
    dim = array.shape

    if dim[1] == 7198:  # all except sub-04
        print(in_fpath[:6], dim, '(before cutting)')
        # in case AO data come before the AV data
        # cut the last 75 TRs from the audio-description's data
        ao = array[:, :3599-75]
        # take all of the movie's data
        av = array[:, 3599:]
        return_array = np.concatenate([ao, av], axis=1)

        new_dim = return_array.shape
        print(in_fpath[:6], new_dim, '(after cutting)')

    elif dim[1] == 7123:  # sub-04
        # correct length -> do noting
        return_array = array
        print(in_fpath[:6], dim, '(unchanged)')

    else:
        raise ValueError('unexpected number of TRs')

    return return_array


def fit_srm(list_of_arrays, out_dir):
    '''Fits the SRM and saves it to files
    '''
    srm = brainiak.funcalign.srm.SRM(features=n_feat, n_iter=n_iter)

    # fit the SRM model
    print(f'fitting SRM to data of all subjects except {subj}...')
    srm.fit(list_of_arrays)

    return srm


def shuffle_all_arrays(all_arrays, timings):
    '''
    '''

    starts = [sum(timings[0:idx+1]) for idx, value in enumerate(timings)]
    starts_ends = [[x, y] for x, y in zip(starts[:-1], starts[1:])]

    # substitute the last index for '-1'
    starts_ends[-1][1] = ''

    shuffled_subjs = []
    for subject, array in enumerate(all_arrays):
        print(f'\nSubject {subject}:')
        random.shuffle(starts_ends)

        shuffled_blocks_arrays = []
        for start, end in starts_ends:
            print(start, end)
            # append the current block
            if end:
                shuffled_blocks_arrays.append(array[:, start:end])
            else:
                shuffled_blocks_arrays.append(array[:, start:])

        # manipulate the array
        shuffled_blocks = np.concatenate(shuffled_blocks_arrays, axis=1)
        # concatenate the blocks
        shuffled_subjs.append(shuffled_blocks)

    import ipdb; ipdb.set_trace() #  BREAKPOINT
    return shuffled_subjs


if __name__ == "__main__":
    # read command line arguments
    subj, out_dir, n_feat, n_iter = parse_arguments()

    # find all input files
    aoav_fpathes = find_files(AOAV_TRAIN_PATTERN)
    # filter for non-current subject
    aoav_fpathes = [fpath for fpath in aoav_fpathes if subj not in fpath]

    # find all input files
    vis_fpathes = find_files(VIS_TRAIN_PATTERN)
    # filter for non-current subject
    vis_fpathes = [fpath for fpath in vis_fpathes if subj not in fpath]

    # a) SRM with data from AO & AV
    model = 'srm-ao-av'
    print('\nProcessing data for model', model)

    aoav_arrays = []
    # loops through subjects (one concatenated / masked time-series per sub)
    for aoav_fpath in aoav_fpathes:
        # 75 TRs of run-8 in sub-04 are missing
        ### cutting of all other arrays
        ### not necessarry anymore 'cause TRs got cutted in the script
        ### that performed the masking but still here as 'sanity check'
        corrected_aoav_array = array_cutting(aoav_fpath)

        # perform zscoring across concatenated experiments
        zscored_aoav_array = stats.zscore(corrected_aoav_array,
                                          axis=1,
                                          ddof=1)

        # append to the list of arrays (containing all subjects' arrays)
        aoav_arrays.append(zscored_aoav_array)

    # fit the SRM model
    aoav_srm = fit_srm(aoav_arrays, out_dir)

    # prepare saving results as pickle
    out_file = f'{model}_feat{n_feat}-iter{n_iter}.npz'
    out_fpath = os.path.join(out_dir, subj, out_file)
    # create (sub)directories
    os.makedirs(os.path.dirname(out_fpath), exist_ok=True)
    # save it
    aoav_srm.save(out_fpath)
    print('SRM (AO & AV) saved to', out_fpath)

    # b) SRM with data from AO, AV, VIS
    model = 'srm-ao-av-vis'
    print('\nProcessing data for model', model)
    # extend the data of AO & AV with the VIS data
    aoavvis_arrays = []
    # loops through subjects (one concatenated / masked time-series per sub)
    for aoav_fpath, vis_fpath in zip(aoav_fpathes, vis_fpathes):
        # load the data of AO & AV (again; not time-efficient but what ever)
        # 75 TRs of run-8 in sub-04 are missing
        ###  cutting of all other arrays
        ### not necessarry anymore 'cause TRs got cutted in the script
        ### that performed the masking but still here as 'sanity check'
        corrected_aoav_array = array_cutting(aoav_fpath)

        # load the VIS data
        print('loading', vis_fpath)
        vis_array = np.load(vis_fpath)
        dim = vis_array.shape
        print(vis_fpath[:6], dim)
        # concat AOAV data and VIS data
        aoavvis_array = np.concatenate([corrected_aoav_array, vis_array],
                                       axis=1)

        # perform zscoring across concatenated experiments
        zscored_aoavvis_array = stats.zscore(aoavvis_array,
                                             axis=1,
                                             ddof=1)

        # append to the list of arrays (containing all subjects' arrays)
        aoavvis_arrays.append(zscored_aoavvis_array)

    # fit the SRM model
    aoavvis_srm = fit_srm(aoavvis_arrays, out_dir)

    # prepare saving results as pickle
    out_file = f'{model}_feat{n_feat}-iter{n_iter}.npz'
    out_fpath = os.path.join(out_dir, subj, out_file)
    # create (sub)directories
    os.makedirs(os.path.dirname(out_fpath), exist_ok=True)
    # save it
    aoavvis_srm.save(out_fpath)
    print('SRM (AO, AV, VIS) saved to', out_fpath)

    # c) SRM with shuffled AO & AV data (negative control):
    model = 'srm-ao-av-shuffled'
    print('\nProcessing data for model', model)

    # shuffle the arrays before fitting the model
    # always take the same seed for every subject
    # by deriving it from the subject's number
    random.seed(int(subj[-2:]))

    aoTimings = [0, 451, 441, 438, 488, 462, 439, 542, 338-75]
    aoAvTimings = aoTimings + aoTimings[1:-1] + [338]  # add AV to AO

    shuffled_aoav_arrays = shuffle_all_arrays(aoav_arrays, aoAvTimings)

    # fit the SRM model
    shuffled_aoav_srm = fit_srm(shuffled_aoav_arrays, out_dir)

    # prepare saving results as pickle
    out_file = f'{model}_feat{n_feat}-iter{n_iter}.npz'
    out_fpath = os.path.join(out_dir, subj, out_file)
    # create (sub)directories
    os.makedirs(os.path.dirname(out_fpath), exist_ok=True)
    # save it
    shuffled_aoav_srm.save(out_fpath)
    print('SRM (AO & AV shuffled) saved to', out_fpath)

    # d) SRM with shuffled AO, AV, VIS data (negative control):
    model = 'srm-ao-av-vis-shuffled'
    print('\nProcessing data for model', model)
    # shuffle the arrays before fitting the model
    # always take the same seed for every subject
    # by deriving it from the subject's number
    random.seed(int(subj[-2:]))

    aoTimings = [0, 451, 441, 438, 488, 462, 439, 542, 338-75]
    aoAvTimings = aoTimings + aoTimings[1:-1] + [338]  # add AV to AO
    aoAvVisTimings = aoAvTimings + 4 * [156]

    shuffled_aoavvis_arrays = shuffle_all_arrays(aoavvis_arrays,
                                                 aoAvVisTimings)

    # fit the SRM model
    shuffled_aoavvis_srm = fit_srm(shuffled_aoavvis_arrays, out_dir)
    # prepare saving results as pickle
    out_file = f'{model}_feat{n_feat}-iter{n_iter}.npz'
    out_fpath = os.path.join(out_dir, subj, out_file)
    # create (sub)directories
    os.makedirs(os.path.dirname(out_fpath), exist_ok=True)
    # save it
    shuffled_aoavvis_srm.save(out_fpath)
    print('SRM (AO, AV, VIS shuffled) saved to', out_fpath)

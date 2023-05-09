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

TIMINGS = [451, 441, 438, 488, 462, 439, 542, 338-75]



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

    parser.add_argument('-rseed',
                        required=False,
                        default='0',
                        help='seed for the shuffling of runs')

    args = parser.parse_args()

    sub = args.sub
    outdir = args.outdir
    n_feat = int(args.nfeat)
    n_iter = int(args.niter)
    rseed = int(args.rseed)

    return sub, outdir, n_feat, n_iter, rseed


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


def shuffle_all_arrays(all_arrays, timings):
    '''
    '''
    aoTimings = [0] + TIMINGS
    aoavTimings = aoTimings + aoTimings[1:-1] + [338]  # add AV to AO
    aoavvisTimings = aoavTimings + 4 * [156]

    # make a list of lists with start-end indices per segment
    starts = [sum(aoavvisTimings[0:idx+1]) for idx, value
              in enumerate(aoavvisTimings)]
    starts_ends = [[x, y] for x, y in zip(starts[:-1], starts[1:])]
    # substitute the last index for '-1'
    starts_ends[-1][1] = ''

    shuffled_subjs = []
    for subject, array in enumerate(all_arrays):
        print(f'\nSubject {subject}:')

        # shuffle each paradigm separately
        shuffledAO = random.sample(starts_ends[0:8], 8)
        shuffledAV = random.sample(starts_ends[8:16], 8)
        shuffledVIS = random.sample(starts_ends[16:], 4)
        shuffled_starts_ends = shuffledAO + shuffledAV + shuffledVIS

        shuffled_blocks_arrays = []
        for start, end in shuffled_starts_ends:
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

    return shuffled_subjs


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
    subj, out_dir, n_feat, n_iter, rseed = parse_arguments()

    print(f'Processing {subj} to fit SRM model ' \
          f'of {n_feat} features (iterations={n_iter}, rs={rseed}).')

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

    print(f'fitting model for {subj}')

    if rseed == 0:
        # use always the same seed drawn from subject number
        random.seed(int(subj[-2:]))
        # fit the SRM
        srm = fit_srm(aoavvis_arrays, out_dir)
        # prepared name of output file
        out_file = f'srm-ao-av-vis_feat{n_feat}-iter{n_iter}.npz'
    else:
        # set the seed
        random.seed(rseed)
        # shuffle the runs for each paradigm and subject separately
        shuffled_aoavvis_arrays = shuffle_all_arrays(aoavvis_arrays,
                                                     TIMINGS)
        # fit the SRM to shuffled runs
        srm_aoavvis = fit_srm(shuffled_aoavvis_arrays, out_dir)
        # prepared name of output file
        out_file = f'srm-ao-av-vis_shuffled_feat{n_feat}-iter{n_iter}-{rseed:04d}.npz'

    # create name of output path
    out_fpath = os.path.join(out_dir, subj, 'models', out_file)
    # create (sub)directories
    os.makedirs(os.path.dirname(out_fpath), exist_ok=True)
    # save it
    srm.save(out_fpath)
    print('SRM saved to', out_fpath)

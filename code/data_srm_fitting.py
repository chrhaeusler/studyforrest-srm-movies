#!/usr/bin/env python3
'''
created on Mon March 29th 2021
author: Christian Olaf Haeusler
'''

from glob import glob
from nilearn.input_data import NiftiMasker, MultiNiftiMasker
import argparse
import nibabel as nib
import numpy as np
import os
import re


# constants
IN_PATTERN = 'sub-??/sub-??_task_aomovie-avmovie_run-1-8_bold-filtered.npy'


def parse_arguments():
    '''
    '''
    parser = argparse.ArgumentParser(
        description='masks and concatenates runs of AO & AV stimulus'
    )

    parser.add_argument('-sub',
                        required=False,
                        default='sub-01',
                        help='subject to leave out (e.g. "subj-01")')

    parser.add_argument('-outdir',
                        required=False,
                        default='test',
                        help='output directory (e.g. "sub-01")')

    args = parser.parse_args()

    sub = args.sub
    outdir = args.outdir

    return sub, outdir


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
    subj, out_dir = parse_arguments()
    # save model as (zipped) pickle variable

    # find all input files
    in_fpathes = find_files(IN_PATTERN)
    # filter for non-current subject
    # in_fpathes = [fpath for fpath in in_fpathes if subj not in fpath]

    arrays = []
    for in_fpath in in_fpathes:
        array = np.load(in_fpath)
        dim = array.shape

        # check (hard coded) expected number of TRs
        if dim[1] == 7198:
            arrays.append(array)
            print(in_fpath[:6], dim)
        else:
            # do some zero padding for sub-04 who has ~ 75 TRs missing in
            # run 8
            zero_padded = np.zeros([array.shape[0], 7198], dtype=np.float32)
            print(in_fpath[:6], dim, '(before)')
            zero_padded[:array.shape[0], :array.shape[1]] = array
            print(in_fpath[:6], zero_padded.shape, '(after)')
            arrays.append(zero_padded)

    movie_data = np.concatenate(arrays, axis=0)

    features = 10
    n_iter = 20

    # Create the SRM object

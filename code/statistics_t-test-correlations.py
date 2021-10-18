#!/usr/bin/env python3
'''
created on Mon March 29th 2021
author: Christian Olaf Haeusler
'''

from collections import OrderedDict
from glob import glob
from nilearn.input_data import NiftiMasker, MultiNiftiMasker
from scipy import stats

import argparse
import copy
import ipdb
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import os
import pandas as pd
import re
import subprocess
import sys
sys.path.append('code')
import corrstats

def parse_arguments():
    '''
    '''
    parser = argparse.ArgumentParser(
        description='loads data from VIS runs and denoises them'
    )

    parser.add_argument('-indir',
                        required=False,
                        default='test',
                        help='input directory')

    parser.add_argument('-run',
                        required=False,
                        default='1',
                        help='run to be tested')

    args = parser.parse_args()

    indir = args.indir
    run = args.run

    return indir, run


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
    in_dir, run = parse_arguments()

    stimulus, stim = 'movie', 'AV'  # vs. 'audio-description', 'AO'
    runA = 7
    runB = 8

    df = pd.read_csv('test/corr-empVIS-vs-func.csv')

    corrsAna = df.loc[df['prediction from'] == 'anatomy', 'Pearson\'s r']
    corrsAna = corrsAna.values
    # perform Fisher's z-transformation
    corrsAnaZ = np.arctanh(corrsAna)

    # get corrleation between empirical and prediction from CMS
    allRuns = df.loc[df['prediction from'] == stimulus, ['runs', 'Pearson\'s r']]

    for runA in list(range(1, 8)):
        runB = runA + 1

        # get correlations functional vs. empirical for run A
        corrsRunA = allRuns.loc[allRuns['runs'] == runA, 'Pearson\'s r']
        corrsRunA = corrsRunA.values
        # perform Fisher's z-transformation
        corrsRunAZ = np.arctanh(corrsRunA)

        # compare run 1 (and just run 1) to correlation anatomy vs. empirical
        if runA == 1:
            print(f'Run {runA} vs Anatomy:')
            # independent t-test
            tValue, pValue = stats.ttest_ind(corrsRunAZ, corrsAnaZ)
            tValue = round(tValue, 2)
            pValue = round(pValue, 4)
            print(f'independent:\tt={tValue}, p={pValue}')
            # dependent t-test
            tValue, pValue = stats.ttest_rel(corrsRunAZ, corrsAnaZ)
            tValue = round(tValue, 2)
            pValue = round(pValue, 4)
            print(f'dependent:\tt={tValue}, p={pValue}\n')

        # get correlations functional vs. empirical for run B
        corrsRunB = allRuns.loc[allRuns['runs'] == runB, 'Pearson\'s r']
        corrsRunB = corrsRunB.values
        # perform Fisher's z-transformation
        corrsRunBZ = np.arctanh(corrsRunB)

        print(f'Run {runA} vs {runB}:')
        # independent t-test
        tValue, pValue = stats.ttest_ind(corrsRunAZ, corrsRunBZ)
        tValue = round(tValue, 2)
        pValue = round(pValue, 5)
        print(f'independent:\tt={tValue}, p={pValue}')
        # dependent t-test
        tValue, pValue = stats.ttest_rel(corrsRunAZ, corrsRunBZ)
        tValue = round(tValue, 2)
        pValue = round(pValue, 5)
        print(f'dependent:\tt={tValue}, p={pValue}\n')

        # get correlations between predictions from CMS and anatomy
#         allRuns = df.loc[df['prediction from'] == f'{stim}-vs-anat', ['runs', 'Pearson\'s r']]
#         corrsFunAnaRunA = allRuns.loc[allRuns['runs'] == runA, 'Pearson\'s r']
#         corrsFunAnaRunA = corrsFunAnaRunA.values
#
#         allRuns = df.loc[df['prediction from'] == f'{stim}-vs-anat', ['runs', 'Pearson\'s r']]
#         corrsFunAnaRunB = allRuns.loc[allRuns['runs'] == runB, 'Pearson\'s r']
#         corrsFunAnaRunB = corrsFunAnaRunB.values

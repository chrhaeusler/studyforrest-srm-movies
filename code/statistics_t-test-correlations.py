#!/usr/bin/env python3
'''
created on Mon March 29th 2021
author: Christian Olaf Haeusler

To do:
    - add header to csv
    - add stat and p-value for tests of normality
    - use floats, not scientific annotation when writing to file
    - test vs. Cronbach's?
'''


import argparse
import csv
import numpy as np
import os
import pandas as pd

from glob import glob
from statsmodels.stats.diagnostic import lilliefors
from scipy import stats


# beautifully hardcoded
todoDict = {
    'corr_vis-ppa-vs-estimation_srm-ao-av-vis_feat10.csv':
    [
        ('movie', 1, 'anatomical alignment', 0),
        ('movie', 1, 'visual localizer', 4),
        ('movie', 1, 'movie', 2),
        ('movie', 2, 'movie', 3),
        ('audio-description', 1, 'visual localizer', 4)
    ],
    'corr_av-ppa-vs-estimation_srm-ao-av-vis_feat10.csv':
    [
        ('movie', 1, 'anatomical alignment', 0),
        ('movie', 1, 'visual localizer', 4),
        ('movie', 1, 'movie', 2),
        ('movie', 2, 'movie', 3),
        ('audio-description', 1, 'visual localizer', 4)
    ],
    'corr_ao-ppa-vs-estimation_srm-ao-av-vis_feat10.csv':
    [
        ('movie', 1, 'anatomical alignment', 0)
    ]
}


def parse_arguments():
    '''
    '''
    parser = argparse.ArgumentParser(
        description='reads cvs\'s comprising the correlations between' \
        'empirical and estimated Z-maps'
    )

    parser.add_argument('-indir',
                        required=False,
                        default='test',
                        help='input directory')

    parser.add_argument('-outdir',
                        required=False,
                        default='test',
                        help='ouput directory')

    args = parser.parse_args()

    indir = args.indir
    outdir = args.outdir

    return indir, outdir


def test_shapiro(values):
    '''
    '''
    alpha = 0.05

    print('Shapiro-Wilk-Test:')
    stat, p = stats.shapiro(values)
    print('Statistics=%.3f, p=%.3f' % (stat, p))

    if p > alpha:
        print('Sample looks Gaussian (fail to reject H0)')
    else:
        print('Sample does NOT look Gaussian (reject H0) <-----------')

    return stat, p


def test_lilliefors(values):
    '''
    '''
    alpha = 0.05

    print('Lilliefors:')
    stat, p = lilliefors(values, dist='norm', pvalmethod='table')
    print('Statistics=%.3f, p=%.3f' % (stat, p))

    if p > alpha:
        print('Sample looks Gaussian (fail to reject H0)')
    else:
        print('Sample does NOT look Gaussian (reject H0) <-----------')

    return stat, p


if __name__ == "__main__":
    # read command line arguments
    inDir, outDir = parse_arguments()

    #
    allToWrite = []

    # loop over the input files (one for each paradigm to be estimated)
    for criterion in list(todoDict.items()):
        inFile = criterion[0]
        toDos = criterion[1]

        if 'corr_vis-ppa' in inFile:
            crit = 'vis'
        elif 'corr_av-ppa' in inFile:
            crit = 'av'
        elif 'corr_ao-ppa' in inFile:
            crit = 'ao'
        # change to raising an exception
        else:
            print('unknown criterium')
            break

        inFpath = os.path.join(inDir, inFile)
        print(f'\nreading {inFpath}')
        df = pd.read_csv(inFpath)

        # loop over the pairs to test
        for toDo in toDos:
            currentTestToWrite = []
            # unpack values (explicitly) from dict
            firstPred, firstQuant, secondPred, secondQuant = toDo
            print(f'\nTesting {firstPred} ({firstQuant} runs) vs. '
                  f'{secondPred} ({secondQuant} runs)')

            # filter for first predictor
            firstDf = df.loc[df['prediction via'] == firstPred,
                            ['number of runs', 'Pearson\'s r']]
            # filter for first predictor's number of runs / segments
            firstVals = firstDf.loc[firstDf['number of runs'] == firstQuant,
                                    'Pearson\'s r'].values

            # filter for second predictor
            secondDf = df.loc[df['prediction via'] == secondPred,
                            ['number of runs', 'Pearson\'s r']]
            # filter for second predictor's number of runs / segments
            secondVals = secondDf.loc[secondDf['number of runs'] == secondQuant,
                                    'Pearson\'s r'].values

            # Do Fisher's z-transformation
            firstValsZ = np.arctanh(firstVals)
            secondValsZ = np.arctanh(secondVals)

            print('\nTests for normality:')


            print(f'{firstPred} ({firstQuant} runs), Fisher\'s transformed')
            test_shapiro(firstValsZ)
            test_lilliefors(firstValsZ)

            print(f'{secondPred} ({secondQuant} runs), Fisher\'s transformed')
            test_shapiro(secondValsZ)
            test_lilliefors(secondValsZ)

            # perform the t-tests
            ttalpha = 0.05
            print(f'\nt-test {firstPred} ({firstQuant} runs) vs. {secondPred} ({secondQuant})')

            # independent t-test
            indtValue, indpValue = stats.ttest_ind(firstValsZ, secondValsZ)
            if indpValue <= ttalpha:
                print(f'independent:\tt={indtValue:.4f}, p={indpValue:.4f}\tsignficiant')
            else:
                print(f'independent:\tt={indtValue:.4f}, p={indpValue:.4f}\tNOT signficiant')

            # dependent t-test
            deptValue, deppValue = stats.ttest_rel(firstValsZ, secondValsZ)
            if deppValue <= ttalpha:
                print(f'dependent:\tt={deptValue:.4f}, p={deppValue:.4f}\tsignficiant')
            else:
                print(f'dependent:\tt={deptValue:.4f}, p={deppValue:.4f}\tNOT signficiant')

            currentTestToWrite = [crit,
                                  firstPred, firstQuant,
                                  secondPred, secondQuant,
                                  indtValue, indpValue,
                                  deptValue, deppValue
                                  ]

            allToWrite.append(currentTestToWrite)

    # write results to file
    outFpath = os.path.join(outDir, 'statistics_t-tests.csv')
    with open(outFpath, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(allToWrite)

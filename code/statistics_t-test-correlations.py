#!/usr/bin/env python3
'''
author: Christian Olaf Haeusler
'''


import argparse
import csv
import numpy as np
import os
import pandas as pd
from statsmodels.stats.diagnostic import lilliefors
from scipy import stats


# beautifully hardcoded
todoDict = {
    'corr_vis-ppa-vs-estimation_srm-ao-av-vis_feat10.csv':
    [
        ('visual localizer', 3, 'anatomical alignment', 0),
        ('movie', 1, 'anatomical alignment', 0),
        ('movie', 1, 'visual localizer', 3),
        ('movie', 2, 'movie', 1),
        ('movie', 3, 'movie', 2),
        ('audio-description', 8, 'anatomical alignment', 0)
    ],
    'corr_av-ppa-vs-estimation_srm-ao-av-vis_feat10.csv':
    [
        ('movie', 1, 'anatomical alignment', 0),
        ('visual localizer', 3, 'anatomical alignment', 0),
        ('movie', 1, 'visual localizer', 3),
        ('movie', 2, 'movie', 1),
        ('movie', 3, 'movie', 2),
        ('audio-description', 8, 'anatomical alignment', 0)

    ],
    'corr_ao-ppa-vs-estimation_srm-ao-av-vis_feat10.csv':
    [
        ('audio-description', 1, 'anatomical alignment', 0),
        ('audio-description', 8, 'anatomical alignment', 0),
        ('movie', 8, 'anatomical alignment', 0)
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
                        default='results',
                        help='input directory')

    parser.add_argument('-outdir',
                        required=False,
                        default='results',
                        help='ouput directory')

    args = parser.parse_args()

    indir = args.indir
    outdir = args.outdir

    return indir, outdir


def test_normality(values, test='shapiro', alphalvl=0.05):
    '''
    '''
    if test == 'shapiro':
        print('Shapiro-Wilk-Test:')
        stat, p = stats.shapiro(values)
    elif test == 'lilliefors':
        print('Lilliefors:')
        stat, p = lilliefors(values, dist='norm', pvalmethod='table')

    print('Statistics=%.3f, p=%.3f' % (stat, p))

    if p > alphalvl:
        print('Sample looks Gaussian (fail to reject H0)')
    else:
        print('Sample does NOT look Gaussian (reject H0) <-----------')

    return '{:.7f}'.format(stat), '{:.7f}'.format(p)


def test_ttest(sampleOne, sampleTwo, alphalvl=0.05, test='dependent'):
    '''
    '''
    if test == 'dependent':
        # dependent t-test
        t, p = stats.ttest_rel(sampleOne, sampleTwo)
    elif test == 'independent':
        t, p = stats.ttest_ind(sampleOne, sampleTwo)

    if p <= alphalvl:
        print(f'{test} t-test:\tt={t:.7f}, p={p:.7f}\tsignificant')
    else:
        print(f'{test} t-test:\tt={t:.7f}, p={p:.7f}\tNOT significant')

    return '{:.7f}'.format(t), '{:.7f}'.format(p)


if __name__ == "__main__":
    # read command line arguments
    inDir, outDir = parse_arguments()

    noOfTests = sum(([len(tests) for tests in todoDict.values()]))
    alpha = 0.05 / noOfTests

    print(f'Number of tests: {noOfTests}; adjusted a: {alpha}')

    allToWrite = []
    # loop over the input files (one for each paradigm to be estimated)
    for criterion in list(todoDict.items()):
        inFile = criterion[0]
        toDos = criterion[1]

        # change to regular expression
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
            print(f'{firstPred} ({firstQuant} runs) vs. '
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


#             print('Tests for normality:')
#             print(f'{firstPred} ({firstQuant} runs), Fisher\'s transformed')
#             statFirstShap, pFirstShap = test_normality(firstValsZ,
#                                                        test='shapiro')
#             statFirstLill, pFirstLill = test_normality(firstValsZ,
#                                                        test='lilliefors')
#
#             print(f'{secondPred} ({secondQuant} runs), Fisher\'s transformed')
#             statSeconShap, pSeconShap = test_normality(secondValsZ,
#                                                        test='shapiro')
#             statSeconLill, pSeconLill = test_normality(secondValsZ,
#                                                        test='lilliefors')
#
#             # perform the t-tests
#             print(f't-test {firstPred} ({firstQuant} runs)' \
#                   f'vs. {secondPred} ({secondQuant})')
#             # independent t-test
#             indtValue, indpValue = test_ttest(firstValsZ, secondValsZ,
#                                               test='independent')
            # dependent t-test
            deptValue, deppValue = test_ttest(firstValsZ, secondValsZ,
                                              alpha,
                                              test='dependent')

            # chain the variables to be written to file later
            currentTestToWrite = [crit,
                                  firstPred, firstQuant,
                                  secondPred, secondQuant,
#                                   indtValue, indpValue,
                                  deptValue, deppValue,
#                                   statFirstShap, pFirstShap,
#                                   statFirstLill, pFirstLill,
#                                   statSeconShap, pSeconShap,
#                                   statSeconLill, pSeconLill
                                  ]

            # extend current loops to previous loop
            allToWrite.append(currentTestToWrite)

    # write results to file
    outFpath = os.path.join(outDir, 'statistics_t-tests.csv')
    header = [
        'citerion',
        'pred1', 'quant1',
        'pred2', 'quant2',
#         'ind t-test t', 'ind t-test p',
        'dep t-test t', 'dep t-test p',
#         'pred1ShapW', 'pred1ShapP',
#         'pred1LillD', 'pred1LillP',
#         'pred2ShapW', 'pred2ShapP',
#         'pred2LillD', 'pred2LillP',
    ]

    with open(outFpath, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(allToWrite)

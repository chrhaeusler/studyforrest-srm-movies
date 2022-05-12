#!/usr/bin/env python3
'''
created on Mon March 29th 2021
author: Christian Olaf Haeusler
'''

from glob import glob
import argparse
import ipdb
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns


def parse_arguments():
    '''
    '''
    parser = argparse.ArgumentParser(
        description='loads a csv file to plot data as stripplot')

    parser.add_argument('-infile',
                        required=False,
                        default='test/corr-empVIS-vs-func.csv',
                        help='the data as csv')

    parser.add_argument('-outdir',
                        required=False,
                        default='test',
                        help='output directory')

    args = parser.parse_args()

    infile = args.infile
    outdir = args.outdir

    return infile, outdir


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
    inFile, outDir = parse_arguments()



    # close figure
    plt.close()

    df = pd.read_csv(inFile)

    sns.set_theme(style='whitegrid')

    ax = sns.stripplot(x='runs',
                       y="Pearson's r",
                       hue='prediction from',
                       jitter=0.2,
                       linewidth=1,
                       dodge=True,
                       data=df)

    # prepare name of output file
    if 'VIS' in inFile:
        which_PPA = 'VIS'
    elif 'AO' in inFile:
        which_PPA = 'AO'
    else:
        print('unkown predicted PPA (must be VIS or AO)')

    plt.savefig(f'{outDir}/stripplot-{which_PPA}.svg',
                bbox_inches='tight')

    plt.savefig(f'{outDir}/stripplot-{which_PPA}.png',
                bbox_inches='tight')

    plt.show()

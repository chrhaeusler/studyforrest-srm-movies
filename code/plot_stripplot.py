#!/usr/bin/env python3
'''
created on Mon March 29th 2021
author: Christian Olaf Haeusler
'''

from glob import glob
import argparse
import ipdb
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns


def parse_arguments():
    '''
    '''
    parser = argparse.ArgumentParser(
        description='loads csv files to plot data as stripplot')

    parser.add_argument('-invis',
                        required=False,
                        default='test/corr_vis-ppa-vs-estimation_srm-ao-av-vis_feat10.csv',
                        help='the data as csv')

    parser.add_argument('-inav',
                        required=False,
                        default='test/corr_av-ppa-vs-estimation_srm-ao-av-vis_feat10.csv',
                        help='the data as csv')

    parser.add_argument('-inao',
                        required=False,
                        default='test/corr_ao-ppa-vs-estimation_srm-ao-av-vis_feat10.csv',
                        help='the data as csv')

    parser.add_argument('-outdir',
                        required=False,
                        default='test',
                        help='output directory')

    args = parser.parse_args()

    inVisResults = args.invis
    inAvResults = args.inav
    inAoResults = args.inao
    outdir = args.outdir

    return inVisResults, inAvResults, inAoResults, outdir


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


def plot_subplot(axNr, title, df, legend=True):
    '''
    '''
    axes[axNr].set_title(title)

    ax = sns.stripplot(ax=axes[axNr],
                       x='number of runs',
                       y="Pearson's r",
                       hue='prediction via',
                       hue_order=['anatomical alignment',
                                  'visual localizer',
                                  'movie',
                                  'audio-description'
                                  ],
                       palette = {'anatomical alignment' : 'grey',
                                  'visual localizer' : 'limegreen',
                                  'movie' : 'red',
                                  'audio-description' : 'blue'
                                  },
                       jitter=0.2,
                       linewidth=1,
                       dodge=True,
                       data=df)

    if not legend:
        ax.legend().remove()

    return ax


if __name__ == "__main__":
    # read command line arguments
    visResults, avResults, aoResults, outDir = parse_arguments()

    # some preparations for plotting
    # close figure
    plt.close()
    # set style for seaborn
    sns.set_theme(style='whitegrid')
    # set seed to always have the same jitter
    np.random.seed(seed=1984)

    # create figure
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=False)

    # figure title
    fig.suptitle('Correlations between empirical and predicted Z-maps')

    # upper subplot: visual localizer
    # read the data first
    visDf = pd.read_csv(visResults)
    axNr = 0
    axes[axNr] = plot_subplot(axNr,
                              'PPA localized via visual localizer (Sengupta et al., 2016)',
                              visDf,
                              legend=True
                              )

    # plot middle subplot
    avDf = pd.read_csv(avResults)
    axNr = 1
    axes[axNr] = plot_subplot(axNr,
                              'PPA localized via movie (Häusler et al., 2022)',
                              avDf,
                              legend=False
                              )

    # plot lower subplot
    aoDf = pd.read_csv(aoResults)
    axNr = 2
    axes[axNr] = plot_subplot(axNr,
                              'PPA localized via audio-description (Häusler et al., 2022)',
                              aoDf,
                              legend=False
                              )

    plt.subplots_adjust(hspace=0.4,
                        wspace=None,
                        top=None,
                        bottom=None,
                        left=None,
                        right=None
                        )

    plt.savefig(f'{outDir}/plot_corr-emp-vs-estimation.pdf',
                bbox_inches='tight')

    plt.savefig(f'{outDir}/plot_corr-emp-vs-estimation.png',
                bbox_inches='tight')

    plt.savefig(f'{outDir}/plot_corr-emp-vs-estimation.svg',
                bbox_inches='tight')

    plt.show()

    plt.close()

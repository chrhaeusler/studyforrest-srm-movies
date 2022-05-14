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

    parser.add_argument('-invis',
                        required=False,
                        default='test/srm-ao-av-vis_feat10_corr_VIS-PPA-vs-CFS-PPA.csv',
                        help='the data as csv')

    parser.add_argument('-inao',
                        required=False,
                        default='test/srm-ao-av-vis_feat10_corr_AO-PPA-vs-CFS-PPA.csv',
                        help='the data as csv')


    parser.add_argument('-outdir',
                        required=False,
                        default='test',
                        help='output directory')

    args = parser.parse_args()

    inVisResults = args.invis
    inAoResults = args.inao
    outdir = args.outdir

    return inVisResults, inAoResults, outdir


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
    visResults, aoResults, outDir = parse_arguments()

    # close figure
    plt.close()


    sns.set_theme(style='whitegrid')

    # create figure
    fig, axes = plt.subplots(2, 1, figsize=(12,8), sharex=False)

    # figure title
    fig.suptitle('Correlations between empirical and predicted Z-maps')

    # plot upper subplot
    # read the data first
    visDf = pd.read_csv(visResults)
    axNr = 0
    axes[0] = plot_subplot(axNr,
                           'PPA localized via visual localizer (Sengupta et al., 2016)',
                           visDf,
                           legend=True
                           )

    # plot lower subplot
    aoDf = pd.read_csv(aoResults)
    axNr = 1
    axes[1] = plot_subplot(axNr,
                           'PPA localized via audio-description (Häusler et al., 2022)',
                           aoDf,
                           legend=False
                           )

    plt.subplots_adjust(hspace=0.35,
                        wspace=None,
                        top=None,
                        bottom=None,
                        left=None,
                        right=None
                        )

    plt.savefig(f'{outDir}/stripplot.pdf',
                bbox_inches='tight')

    plt.savefig(f'{outDir}/stripplot.png',
                bbox_inches='tight')

    plt.show()

    plt.close()

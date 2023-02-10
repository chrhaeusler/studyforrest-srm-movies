#!/usr/bin/env python3
'''
created on Mon March 29th 2021
author: Christian Olaf Haeusler
'''

import argparse
import ipdb
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import os
import pandas as pd
import seaborn as sns

STIMULICOLORS = {# 'anatomical alignment': 'grey',
                 'visual localizer': 'y',
                 'movie': 'red',
                 'audio-description': 'blue'
                 }


def parse_arguments():
    '''
    '''
    parser = argparse.ArgumentParser(
        description='loads a csv file to plot data as stripplot')

    parser.add_argument('-invis',
                        required=False,
                        default='test/srm-ao-av-vis_feat10_corr_VIS-PPA-vs-CFS-PPA.csv',
                        help='the data as csv')

    parser.add_argument('-inav',
                        required=False,
                        default='test/srm-ao-av-vis_feat10_corr_AV-PPA-vs-CFS-PPA.csv',
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
    inAvResults = args.inav
    inAoResults = args.inao
    outdir = args.outdir

    return inVisResults, inAvResults, inAoResults, outdir


def plot_boxplot(axis, df):
    '''
    '''
    axis = sns.boxplot(
        data=df,
        ax=axis,
        zorder=1,  # boxplots are supposed to be behind the stripplot
        x='stimulus',
        y="Cronbach's a",
        width=0.5,
        palette = STIMULICOLORS,
        # median
        medianprops={'visible': True,
                     'color': 'dimgrey',
                     'ls': '-',
                     'lw': 1,
                     # 'alpha': 0.3
                     },
        # mean
        showmeans=True,
        meanline=True,
        meanprops={'color': 'dimgrey',
                   'ls': (0, (1, 1)),
                   'lw': 1,
                   # 'alpha': 0.3
                   },
        # outliers
        showfliers=False,
        # flierprops={'marker': 'x', 'markersize': 8},
        # box
        showbox=True,
        boxprops={# 'color': STIMULICOLORS.items(),
                  'lw': 1,
                  'alpha': 0.3
                  },
        # whiskers
        whiskerprops={'color': 'dimgrey',
                      'lw': 1,
                      'alpha': 0.3
                      },
        # caps (horizontal lines at the ends of whiskers)
        showcaps=True,
        capprops={'color': 'dimgrey',
                  'alpha': 0.3
                  },
    )

    # remove the handles and labels from the legend
    # handles, labels = axis.get_legend_handles_labels()
    # axis.legend([], [], loc='lower right')

    return axis


def plot_stripplot(axis, df):
    '''
    '''
    ax = sns.stripplot(ax=axis,
                       data=df,
                       x='stimulus',
                       y="Cronbach's a",
                       palette = STIMULICOLORS,
                       jitter=0.2,
                       linewidth=1,
                       dodge=True
                       )

    # rename the x axis label
    # ax.set_xlabel('paradigm')
    # or just writen nothing
    ax.set_xlabel('')
    ax.set_ylabel(r"Cronbach's $\alpha$")


if __name__ == "__main__":
    # for now hard code some stuff
    cronbachsResults = 'test/statistics_cronbachs.csv'
    outDir = 'test'

    # read command line arguments
    # visResults, avResults, aoResults, outDir = parse_arguments()

    df = pd.read_csv(cronbachsResults)

    # some preparations for plotting
    # close figure
    plt.close()
    # set style for seaborn
    sns.set_theme(style='whitegrid')
    # set seed to always have the same jitter
    np.random.seed(seed=1986)

    fig, axis = plt.subplots(1, 1, figsize=(6, 5), sharex=False)

    plot_boxplot(axis, df)
    plot_stripplot(axis, df)

    # define the parameters for the legend
    lineMedian = Line2D([0], [0],
                        label='median',
                        color='dimgrey',
                        linestyle='-',
                        linewidth=1
                        )

    lineMean = Line2D([0], [0],
                      label='mean',
                      color='dimgrey',
                      linestyle=(0, (1, 1)),
                      linewidth=1
                      )

    # plot the legend
    plt.legend(handles=[lineMedian, lineMean], loc='lower left')


    # calculate the medians per stimulus
    means = pd.pivot_table(df, index='stimulus', aggfunc=np.mean)
    medians = pd.pivot_table(df, index='stimulus', aggfunc=np.median)


#     for stimCol in list(STIMULICOLORS.items())[1:]:  # first index is anatomical
#         stimulus = stimCol[0]
#         color = stimCol[1]
#         # plot medians
#         mean = means.loc[stimulus]["Cronbach's a"]
#         plt.axhline(y=mean, color=color,
#                     linestyle=(0, (3, 5, 1, 5, 1, 5)),
#                     alpha=1
#                     )
#
#         median = medians.loc[stimulus]["Cronbach's a"]
#         # handle that median of movie and audio-description are on top of each
#         # others; plot a solid line for the movie (underlying the
#         # audio-description
#         if stimulus == 'movie':
#
#             plt.axhline(y=median, color=color,
#                         linestyle=(3, (3, 3)),
#                         alpha=1)
#
#         else:
#             plt.axhline(y=median, color=color,
#                         linestyle=(0, (3, 3)),
#                         alpha=1
#                         )

# save the figure
    os.makedirs(outDir, exist_ok=True)

    extensions = ['pdf', 'png', 'svg']
    for extension in extensions:
        fpath = os.path.join(outDir, f'plot_cronbachs.{extension}')
        plt.savefig(fpath, bbox_inches='tight')

    plt.show()

    plt.close()

#!/usr/bin/env python3
'''
created on Mon March 29th 2021
author: Christian Olaf Haeusler
'''

from glob import glob
import argparse
import ipdb
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
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


def plot_boxplot(data):
    '''
    '''
    sns.boxplot(
        data=data,
        # ax=axis,
        zorder=2,  # boxplots are supposed to be behind the stripplot
        x='number of voxels',
        width=0.1,
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
        boxprops={'color': 'dimgrey',
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
    plt.legend([], [], loc='lower right')

    # return axis


def plot_stripplot(data):
    '''
    '''
    # do the actual plotting
    sns.stripplot(data=data,
                  x='number of voxels',
                  jitter=0.02,
                  linewidth=1,
                  color='dimgrey',
                  edgecolor='black'
                  )

    # limit the axes
    plt.xlim([1200, 2000])
    plt.ylim([-0.1, 0.1])

    # return ax

if __name__ == "__main__":
    # for now, hard code some stuff
    cronbachsResults = 'test/statistics_cronbachs.csv'
    outDir = 'test'

    # read command line arguments
    # visResults, avResults, aoResults, outDir = parse_arguments()

    df = pd.read_csv(cronbachsResults)
    voxelDf = df.loc[df['stimulus'] == 'visual localizer']

    # some preparations for plotting
    # close figure
    plt.close()
    # set style for seaborn
    sns.set_theme(style='whitegrid')
    # set seed to always have the same jitter
    np.random.seed(seed=1984)

    # create the figure
    plt.figure(figsize=(12, 2))

    # plot the underlying boxplot
    plot_boxplot(voxelDf)

    # plot the stripplot (points per subject)
    plot_stripplot(voxelDf)

    # define the parameters for the legend
    lineSubject = Line2D([0], [0],
                         label='subject',
                         linestyle='None',
                         marker='o',
                         color='dimgrey',
                         markeredgecolor='black'
                         )

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
    plt.legend(handles=[lineMedian, lineMean], loc='center left')

    # use a tight layout
    plt.tight_layout()

    # save the figure
    os.makedirs(outDir, exist_ok=True)

    # save to different file formats
    extensions = ['pdf', 'png', 'svg']
    for extension in extensions:
        fpath = os.path.join(outDir, f'plot_voxel-counts.{extension}')
        plt.savefig(fpath, bbox_inches='tight')

    plt.show()

    plt.close()

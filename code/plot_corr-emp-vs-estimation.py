#!/usr/bin/env python3
'''
created on Mon March 29th 2021
author: Christian Olaf Haeusler
'''

import argparse
import ipdb
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

STIMULICOLORS = {'anatomical alignment': 'grey',
                 'visual localizer': 'limegreen',
                 'movie': 'red',
                 'audio-description': 'blue'
                 }


def parse_arguments():
    '''
    '''
    parser = argparse.ArgumentParser(
        description='loads csv files to plot data as stripplot')

    parser.add_argument('-invis',
                        required=False,
                        default='test/corr_vis-ppa-vs-estimation_srm-ao-av-vis_feat10.csv',
                        help='csv file with correlations VIS vs. estimation')

    parser.add_argument('-inav',
                        required=False,
                        default='test/corr_av-ppa-vs-estimation_srm-ao-av-vis_feat10.csv',
                        help='csv file with correlations AV vs. estimation')

    parser.add_argument('-inao',
                        required=False,
                        default='test/corr_ao-ppa-vs-estimation_srm-ao-av-vis_feat10.csv',
                        help='csv file with correlations AO vs. estimation')

    parser.add_argument('-incronb',
                        required=False,
                        default='test/statistics_cronbachs.csv',
                        help='csv file with Cronbachs Alpha of PPA from VIS, AV & AO')

    parser.add_argument('-outdir',
                        required=False,
                        default='test',
                        help='output directory')

    args = parser.parse_args()

    inVisResults = args.invis
    inAvResults = args.inav
    inAoResults = args.inao
    inCronbachs = args.incronb
    outdir = args.outdir

    return inVisResults, inAvResults, inAoResults, inCronbachs, outdir


def plot_boxplot(axis, df):
    '''
    '''
    axis = sns.boxplot(
        data=df,
        ax=axis,
        zorder=2,  # boxplots are supposed to be behind the stripplot
        x='number of runs',
        y="Pearson's r",
        hue='prediction via',
        hue_order=STIMULICOLORS.keys(),
        palette=STIMULICOLORS,
        # median
        medianprops={'visible': True, 'color': 'k'},
        # mean
        showmeans=True,
        meanline=True,
        meanprops={'color': 'k', 'ls': '--', 'lw': 1, 'alpha': 0.3},
        # outliers
        showfliers=True,
        flierprops={'marker': 'x', 'markersize': 8},
        # box
        showbox=True,
        boxprops={'alpha': 0.3},
        # whiskers
        whiskerprops={'visible': True, 'alpha': 0.3},
        # caps (horizontal lines at the ends of whiskers)
        showcaps=True,
        capprops={'alpha': 0.3},
    )

    # remove the handles and labels from the legend
    # handles, labels = axis.get_legend_handles_labels()
    axis.legend([], [], loc='lower right')

    return axis


def plot_subplot(axis, title, df, boxplot=False, legend=True):
    '''
    '''
    axis.set_title(title)

    if boxplot is True:
        axis = plot_boxplot(axis, df)

    axis = sns.stripplot(
        data=df,
        ax=axis,
        x='number of runs',
        y="Pearson's r",
        hue='prediction via',
        hue_order=STIMULICOLORS.keys(),
        palette=STIMULICOLORS,
        jitter=0.2,
        linewidth=1,
        dodge=True
    )

    # set the same limits for the y axis
    # to make subplots comparable
    axis.set_ylim([-0.3, 1])
    axis.set_yticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0])

    # one legend in the figure is sufficient
    # remove it from (some) subplots
    if legend:
        handles, labels = axis.get_legend_handles_labels()
        axis.legend(handles[-4:], labels[-4:], loc='lower right')
    else:
        axis.legend().remove()

    return axis


if __name__ == "__main__":
    # read command line arguments
    visResults, avResults, aoResults, cronbachs, outDir = parse_arguments()

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

    # adjust some spacings
    plt.subplots_adjust(hspace=0.4,
                        wspace=None,
                        top=None,
                        bottom=None,
                        left=None,
                        right=None
                        )

    # upper subplot: visual localizer
    # read the first
    visDf = pd.read_csv(visResults)
    axNr = 0
    # call the plotting function
    axes[axNr] = plot_subplot(
        axes[axNr],
        'PPA localized via visual localizer (Sengupta et al., 2016)',
        visDf,
        boxplot=True,
        legend=True
    )

    # plot middle subplot
    avDf = pd.read_csv(avResults)
    axNr = 1
    # call the plotting function
    axes[axNr] = plot_subplot(
        axes[axNr],
        'PPA localized via movie (Häusler et al., 2022)',
        avDf,
        boxplot=True,
        legend=False
    )

    # plot lower subplot
    aoDf = pd.read_csv(aoResults)
    axNr = 2
    # call the plotting function
    axes[axNr] = plot_subplot(
        axes[axNr],
        'PPA localized via audio-description (Häusler et al., 2022)',
        aoDf,
        boxplot=True,
        legend=False
    )

    # add the Cronbach's a to the subplot
    # which measure across subjects to calculate?
    centrTendFunc = np.median  # vs. np.mean
    # read the results of Cronbachs per stimulus per subject
    df = pd.read_csv(cronbachs)
    # pivot the table
    centrTends = pd.pivot_table(df, index='stimulus', aggfunc=centrTendFunc)

    # loop over the subplots
    # and add measure of central tendency for predicted PPA
    for axNr, stimCol in enumerate(list(STIMULICOLORS.items())[1:]):
        stimulus = stimCol[0]
        color = stimCol[1]

        # get the value of central tendency of the current loops stimulus
        centrTend = centrTends.loc[stimulus]["Cronbach's a"]
        axes[axNr].axhline(y=centrTend,
                           color=color,
                           linestyle='--',
                           alpha=0.6)

    # save the figure
    os.makedirs(outDir, exist_ok=True)

    extensions = ['pdf', 'png', 'svg']
    for extension in extensions:
        fpath = os.path.join(outDir,
                             f'plot_corr-emp-vs-estimation.{extension}')
        plt.savefig(fpath, bbox_inches='tight')

    plt.show()

    plt.close()

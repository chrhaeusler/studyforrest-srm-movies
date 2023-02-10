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
                 'visual localizer': 'y',
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
        zorder=1,  # boxplots are supposed to be behind the stripplot
        x='number of runs',
        y="Pearson's r",
        hue='prediction via',
        hue_order=STIMULICOLORS.keys(),
        palette=STIMULICOLORS,
        # width=0.1,
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
    axis.legend([], [], loc='lower right')

    return axis


def plot_subplot(axis, title, df, boxplot=False, legend=True, xlabel=True, ylabel=True):
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

    # set a custom label for the x axis
    if xlabel == True:
        axis.set_xlabel('number of runs / segments')
    else:
        x_axis = axis.axes.get_xaxis()
        x_label = x_axis.get_label()
        x_label.set_visible(False)

    if ylabel == False:
        y_axis = axis.axes.get_yaxis()
        y_label = y_axis.get_label()
        y_label.set_visible(False)

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
    # fig.suptitle('Correlations between empirical and predicted Z-maps')

    # adjust some spacings
    plt.subplots_adjust(hspace=0.35,
                        wspace=None,
                        top=None,
                        bottom=None,
                        left=None,
                        right=None
                        )

    # add the Cronbach's a to the subplot
    # read the results of Cronbachs per stimulus per subject
    df = pd.read_csv(cronbachs)
    # pivot the table
    medians = pd.pivot_table(df, index='stimulus', aggfunc=np.median)
    means = pd.pivot_table(df, index='stimulus', aggfunc=np.mean)

    # loop over the subplots
    # and add measure of central tendency for predicted PPA
    for axNr, stimCol in enumerate(list(STIMULICOLORS.items())[1:]):
        stimulus = stimCol[0]
        color = stimCol[1]

        # get the value of central tendency of the current loops stimulus
        median = medians.loc[stimulus]["Cronbach's a"]
        axes[axNr].axhline(y=median,
                           zorder=0,
                           color=color,
                           linestyle='-',
                           alpha=0.4
                           )

        mean = means.loc[stimulus]["Cronbach's a"]
        axes[axNr].axhline(y=mean,
                           zorder=0,
                           color=color,
                           linestyle=(0, (1, 1)),
                           alpha=0.4
                           )

    # upper subplot: visual localizer
    # read the first
    visDf = pd.read_csv(visResults)
    axNr = 0
    # call the plotting function
    axes[axNr] = plot_subplot(
        axes[axNr],
        'Estimation of the visual localizer\'s empirical $\it{Z}$-maps (cf. Sengupta et al., 2016)',
        visDf,
        boxplot=True,
        legend=True,
        ylabel=False,
        xlabel=False
    )

    # plot middle subplot
    avDf = pd.read_csv(avResults)
    axNr = 1
    # call the plotting function
    axes[axNr] = plot_subplot(
        axes[axNr],
        'Estimation of the movie\'s empirical $\it{Z}$-maps (cf. Häusler et al., 2022)',
        avDf,
        boxplot=True,
        legend=False,
        xlabel=False
    )

    # plot lower subplot
    aoDf = pd.read_csv(aoResults)
    axNr = 2
    # call the plotting function
    axes[axNr] = plot_subplot(
        axes[axNr],
        'Estimation of the audio-description\'s empirical $\it{Z}$-maps (cf. Häusler et al., 2022)',
        aoDf,
        boxplot=True,
        legend=False,
        ylabel=False
    )

    # save the figure
    os.makedirs(outDir, exist_ok=True)

    extensions = ['pdf', 'png', 'svg']
    for extension in extensions:
        fpath = os.path.join(outDir,
                             f'plot_corr-emp-vs-estimation.{extension}')
        plt.savefig(fpath, bbox_inches='tight')

    plt.show()

    plt.close()

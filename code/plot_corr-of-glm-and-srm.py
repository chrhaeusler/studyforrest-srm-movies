#!/usr/bin/env python3
'''
created on Fri May 21 2021
author: Christian Olaf Haeusler
'''
from glob import glob
import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import brainiak.funcalign.srm


matplotlib.use('Agg')

AO_USED = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 17, 18]
AO_NAMES = {1: 'body',
            2: 'bpart',
            3: 'fahead',
            4: 'furn',
            5: 'geo',
            6: 'groom',
            7: 'object',
            8: 'se_new',
            9: 'se_old',
            10: 'sex_f',
            11: 'sex_m',
            12: 'vse_new',
            13: 'vse_old',
            14: 'vlo_ch',
            15: 'vpe_new',
            16: 'vpe_old',
            17: 'fg_ad_lrdiff',
            18: 'fg_ad_rms'
            }

AV_USED = [1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14]
AV_NAMES = {1: 'vse_new',
            2: 'vse_old',
            3: 'vlo_ch',
            4: 'vpe_new',
            5: 'vpe_old',
            6: 'vno_cut',
            7: 'se_new (ao)',
            8: 'se_old (ao)',
            9: 'fg_av_ger_lr',
            10: 'fg_av_ger_lr_diff',
            11: 'fg_av_ger_ml',
            12: 'fg_av_ger_pd',
            13: 'fg_av_ger_rms',
            14: 'fg_av_ger_ud'
            }

VIS_USED = [1, 2, 3, 4, 5, 6]
VIS_NAMES = {1: 'body',
             2: 'face',
             3: 'house',
             4: 'object',
             5: 'scene',
             6: 'scramble'
             }


def parse_arguments():
    '''
    '''
    parser = argparse.ArgumentParser(
        description="creates the correlation of convoluted regressors from \
        a subject's 1st lvl results directories (= all single run dirs ")

    parser.add_argument('-ao',
                        default='inputs/studyforrest-ppa-analysis/'\
                        'sub-01/run-1_audio-ppa-grp.feat/design.mat',
                        help='pattern of path/file for 1st lvl (AO) design files')

    parser.add_argument('-vis',
                        default='inputs/studyforrest-data-visualrois/'\
                        'sub-01/run-1.feat/design.mat',
                        help='pattern of path/file for 1st lvl (vis) design files')

    parser.add_argument('-av',
                        default='inputs/studyforrest-ppa-analysis/'\
                        'sub-01/run-1_movie-ppa-grp.feat/design.mat',
                        help='pattern of path/file for 1st lvl (AV) design files')

    parser.add_argument('-model',
                        default='sub-01/srm-ao-av-vis_feat10-iter30.npz',
                        help='the model file')

    parser.add_argument('-o',
                        default='test',
                        help='the output directory for the PDF and SVG file')

    args = parser.parse_args()

    aoExample = args.ao
    avExample = args.av
    visExample = args.vis
    modelFile = args.model
    outDir = args.o

    return aoExample, avExample, visExample, modelFile, outDir


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


def find_design_files(example):
    '''
    '''
    # from example, create the pattern to find design files for all runs
    run = re.search('run-\d', example)
    run = run.group()
    designPattern = example.replace(run, 'run-*')

    # just in case, create substitute random subject for sub-01
    subj = re.search('sub-\d{2}', example)
    subj = subj.group()
    designPattern = designPattern.replace(subj, 'sub-01')

    designFpathes = sorted(glob(designPattern))

    return designFpathes


def load_srm(in_fpath):
    # make np.load work with allow_pickle=True
    # save np.load
    np_load_old = np.load
    # modify the default parameters of np.load
    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
    # np.load = lambda *a: np_load_old(*a, allow_pickle=True)
    # load the pickle file
    srm = brainiak.funcalign.srm.load(in_fpath)
    # change np.load() back to normal
    np.load = np_load_old

    return srm


def plot_heatmap(title, matrix, outFpath, usedRegressors=[]):
    '''
    '''
    # generate a mask for the upper triangle
    mask = np.zeros_like(matrix, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    # set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # custom diverging colormap
    cmap = sns.diverging_palette(220, 10, sep=1, as_cmap=True)

    # draw the heatmap with the mask and correct aspect ratio
    sns_plot = sns.heatmap(matrix, mask=mask,
                           cmap=cmap,
                           square=True,
                           center=0,
                           vmin=-1.0, vmax=1,
                           annot=True, annot_kws={"size": 8, "color": "k"}, fmt='.1f',
                           # linewidths=.5,
                           cbar_kws={"shrink": .6}
                           )

    plt.xticks(rotation=90, fontsize=12)
    plt.yticks(rotation=0, fontsize=12)

    # plt.title(title)

    # coloring of ticklabels
    # x-axis
    if usedRegressors == AO_USED:
#        for x in range(len(srm.s_), len(srm.s_) + len(AO_USED)):
        for x in range(len(AO_USED)):
            plt.gca().get_xticklabels()[x].set_color('blue')  # black = default
        # handle the sum of regressors
        for x in range(len(AO_USED), len(AO_USED) + 1):
            plt.gca().get_xticklabels()[x].set_color('cornflowerblue')  # black = default

    elif usedRegressors == AV_USED:
        for x in range(len(AV_USED)):
            plt.gca().get_xticklabels()[x].set_color('red')  # black = default

    elif usedRegressors == VIS_USED:
        for x in range(len(VIS_USED)):
            plt.gca().get_xticklabels()[x].set_color('y')  # black = default

    # y-axis
    if usedRegressors == AO_USED:
        for y in range(len(AO_USED)):
            plt.gca().get_yticklabels()[y].set_color('blue')  # black = default
        for y in range(len(AO_USED), len(AO_USED) + 1):
            plt.gca().get_yticklabels()[y].set_color('cornflowerblue')  # black = default

    elif usedRegressors == AV_USED:
        for y in range(len(AV_USED)):
            plt.gca().get_yticklabels()[y].set_color('red')  # black = default

    elif usedRegressors == VIS_USED:
        for y in range(len(VIS_USED)):
            plt.gca().get_yticklabels()[y].set_color('y')  # black = default


    # create the output path
    os.makedirs(os.path.dirname(outFpath), exist_ok=True)

    extensions = ['pdf', 'png', 'svg']
    for extension in extensions:
        fpath = os.path.join(f'{out_fpath}.{extension}')
        plt.savefig(fpath, bbox_inches='tight')
    plt.close()


def create_aoDf(aofPathes):
    '''
    factorize / merge this function and the two functions below into one
    '''
    # specify which columns of the design file to use
    # correct for python index starting at 0
    # use every 2nd column because odd numbered columns
    # in the design file are temporal derivatives
    ao_columns = [(x-1) * 2 for x in AO_USED]
    ao_reg_names = [AO_NAMES[x] for x in AO_USED]

    # read the 8 design files and concatenate
    aoDf = pd.concat([pd.read_csv(run,
                                  usecols=ao_columns,
                                  names=ao_reg_names,
                                  skiprows=5, sep='\t')
                      for run in aofPathes], ignore_index=True)

    # add a combination of regressors
    aoDf['geo&groom'] = aoDf['geo'] + aoDf['groom']
    # aoDf['geo&groom&furn'] = aoDf['geo'] + aoDf['groom'] + aoDf['furn']

    return aoDf


def create_avDf(avfPathes):
    '''
    '''
    # specify which columns of the design file to use
    # correct for python index starting at 0
    # use every 2nd column because odd numbered columns
    # in the design file are temporal derivatives
    av_columns = [(x-1) * 2 for x in AV_USED]
    av_reg_names = [AV_NAMES[x] for x in AV_USED]

    # read the 8 design files and concatenate
    avDf = pd.concat([pd.read_csv(run,
                                  usecols=av_columns,
                                  names=av_reg_names,
                                  skiprows=5, sep='\t')
                      for run in avfPathes], ignore_index=True)

    return avDf


def create_visDf(visfPathes):
    '''
    '''
    # specify which columns of the design file to use
    # correct for python index starting at 0
    # use every 2nd column because odd numbered columns
    # in the design file are temporal derivatives
    vis_columns = [(x-1) * 2 for x in VIS_USED]
    vis_reg_names = [VIS_NAMES[x] for x in VIS_USED]

    # read the 4 design files and concatenate
    visDf = pd.concat([pd.read_csv(run,
                                   usecols=vis_columns,
                                   names=vis_reg_names,
                                   skiprows=5, sep='\t')
                       for run in visfPathes], ignore_index=True)

    return visDf


def create_srmDf(modelFile):
    '''
    '''
    # load the SRM from file
    srm = load_srm(modelFile)
    # slice SRM model for the TRs of the audio-description
    srm_array = srm.s_.T

    # create pandas dataframe from array and name the columns
    columns = ['shared feature %s' % str(int(x)+1) for x in range(srm_array.shape[1])]

    srmDf = pd.DataFrame(data=srm_array,
                         columns=columns)

    return srmDf


def create_corr_matrix(df1, df2, arctanh=False):
    '''
    '''
    # concat regressors and shared responses
    # slice the dataframe cause the last 75 TRs are not in the model space
    regressorsAndModelDf = pd.concat([df1, df2], axis=1)
    # create the correlation matrix for all columns

    if arctanh is True:
        regCorrMat = regressorsAndModelDf.corr()
        regCorrMat = np.arctanh(regCorrMat)
    else:
        regCorrMat = regressorsAndModelDf.corr()

    return regCorrMat


def handle_one_or_list_of_models(regressorsDf, modelFile, start, end):
    '''it's the thought that counts (and that the function does what it does)
    '''
    if 'shuffled' not in modelFile:
        # read the model file
        srmDf = create_srmDf(modelFile)
        # reset index of SRM's df
        srm_ao_TRs = srmDf[start:end]
        srm_ao_TRs.reset_index(inplace=True, drop=True)

        # concat regressors and shared responses
        # slice the dataframe cause the last 75 TRs are not in the model space
        regCorrMat = create_corr_matrix(regressorsDf, srm_ao_TRs,
                                        arctanh=False)

    else:
        # find all files according to pattern
        directory = os.path.dirname(modelFile)
        modelPattern = model[:-4] + '*.npz'
        modelPattern = os.path.join(directory, modelPattern)
        shuffledModelFpathes = find_files(modelPattern)

        # read in the files and sum up all cells
        for idx, shuffledModelFile in enumerate(shuffledModelFpathes, 1):

            if idx == 1:
                print(f'reading model no. {idx}:', shuffledModelFile)
                # read the model file
                srmDf = create_srmDf(shuffledModelFile)
                # reset index of SRM's df
                srm_ao_TRs = srmDf[start:end]
                srm_ao_TRs.reset_index(inplace=True, drop=True)

                # concat regressors and shared responses
                regCorrMat = create_corr_matrix(regressorsDf, srm_ao_TRs,
                                                arctanh=True)

            else:
                # read the model file
                print(f'reading model no. {idx}:', shuffledModelFile)
                # reset index of SRM's df
                srm_ao_TRs = srmDf[start:end]
                srm_ao_TRs.reset_index(inplace=True, drop=True)

                # concat regressors and shared responses
                regCorrMat += create_corr_matrix(regressorsDf, srm_ao_TRs,
                                                 arctanh=True)

        # take the mean
        regCorrMat = regCorrMat / idx
        regCorrMat = np.tanh(regCorrMat)

    return regCorrMat


if __name__ == "__main__":
    # get the command line inputs
    aoExample, avExample, visExample, modelFile, outDir = parse_arguments()

    # infere subject number form file name
    sub = re.search('sub-\d{2}', modelFile)
    sub = sub.group()

    model = os.path.basename(modelFile).split('.npz')[0]

    # a) plot the correlation of AO regressors and features (only AO TRs)
    # get design.mat files for the 8 runs
    aofPathes = find_design_files(aoExample)
    # create the dataframe
    aoDf = create_aoDf(aofPathes)
    # indices of audio-decription TRs within the model
    start = 0
    end = 3524

    # get the correlation matrix
    regCorrMat = handle_one_or_list_of_models(aoDf, modelFile, start, end)

    # plot it
    title = f'{sub}: AO Regressors vs. Shared Features'\
        ' ({model}; TRs {start}-{end})'
    # create name of path and file (must not include ".{extension}"
    out_fpath = os.path.join(
        outDir, f'corr_ao-regressors-vs-cfs_{sub}_{model}_{start}-{end}')

    plot_heatmap(title, regCorrMat, out_fpath, AO_USED)

    # b) plot the correlation of AV regressors and features (only AV TRs)
    # get design.mat files for the 8 runs
    avfPathes = find_design_files(avExample)
    # create the dataframe
    avDf = create_avDf(avfPathes)
    # indices of movie TRs within the model
    start = 3524
    end = 7123

    # get the correlation matrix
    regCorrMat = handle_one_or_list_of_models(avDf, modelFile, start, end)

    # plot it
    title = f'{sub}: AV Regressors vs. Shared Responses' \
        ' ({model}; TRs {start}-{end})'
    # create name of path and file (must not include file extension)"
    out_fpath = os.path.join(
        outDir, f'corr_av-regressors-vs-cfs_{sub}_{model}_{start}-{end}'
    )

    plot_heatmap(title, regCorrMat, out_fpath, AV_USED)

    # c) plot the correlation of VIS regressors and features (only VIS TRs)
    # get design.mat files for the 4 runs
    visfPathes = find_design_files(visExample)
    # create the dataframe
    visDf = create_visDf(visfPathes)

    # indices of localizer TRs within the model
    start = 7123
    end = 7747

    # get the correlation matrix
    regCorrMat = handle_one_or_list_of_models(visDf, modelFile, start, end)

    # plot it
    title = f'{sub}: VIS Regressors vs. Shared Responses' \
        ' ({model}; TRs {start}-{end})'
    # create name of path and file (must not include file extension)
    out_fpath = os.path.join(
        outDir, f'corr_vis-regressors-vs-cfs_{sub}_{model}_{start}-{end}'
    )

    plot_heatmap(title, regCorrMat, out_fpath, VIS_USED)

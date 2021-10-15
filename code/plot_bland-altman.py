#!/usr/bin/env python2
# -*- coding: utf-8 -*-
'''
author: Christian Olaf Häusler
created on Wednesday September 15 2021

start ipython2.7 via
'python2 -m IPython'
'''

from __future__ import print_function
from collections import OrderedDict
from glob import glob
from mvpa2.datasets.mri import fmri_dataset
from nilearn.input_data import NiftiMasker, MultiNiftiMasker
import argparse
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
import re
import subprocess
# from mvpa2.datasets.mri import fmri_dataset
from scipy.stats import gaussian_kde


# constants
GRP_PPA_PTTRN = 'sub-??/masks/in_bold3Tp2/grp_PPA_bin.nii.gz'
AO_FOV_MASK = 'sub-??/masks/in_bold3Tp2/audio_fov.nii.gz'
GM_MASK = 'sub-??/masks/in_bold3Tp2/gm_bin_dil_fov.nii.gz'

# binary mask(s) of individual visual localizer (in subject space)
PPA_MASK_PATTERN = 'inputs/studyforrest-data-visualrois/'\
    'sub-*/rois/?PPA_?_mask.nii.gz'

# individual 2nd level results (primary cope in subject space)
PREDICTED_PREFIX = 'predicted-VIS-PPA_from_'
PREDICTIONS = [
    'anatomy.nii.gz',
    'srm-ao-av-vis_feat10_0-451.nii.gz',
    'srm-ao-av-vis_feat10_0-892.nii.gz',
    'srm-ao-av-vis_feat10_0-1330.nii.gz',
    'srm-ao-av-vis_feat10_0-1818.nii.gz',
    'srm-ao-av-vis_feat10_0-2280.nii.gz',
    'srm-ao-av-vis_feat10_0-2719.nii.gz',
    'srm-ao-av-vis_feat10_0-3261.nii.gz',
    'srm-ao-av-vis_feat10_0-3524.nii.gz',
    'srm-ao-av-vis_feat10_3524-3975.nii.gz',
    'srm-ao-av-vis_feat10_3524-4416.nii.gz',
    'srm-ao-av-vis_feat10_3524-4854.nii.gz',
    'srm-ao-av-vis_feat10_3524-5342.nii.gz',
    'srm-ao-av-vis_feat10_3524-5804.nii.gz',
    'srm-ao-av-vis_feat10_3524-6243.nii.gz',
    'srm-ao-av-vis_feat10_3524-6785.nii.gz',
    'srm-ao-av-vis_feat10_3524-7123.nii.gz',
    'srm-ao-av-vis_feat10_0-7123.nii.gz'
]


VIS_ZMAP_PATTERN = 'inputs/studyforrest-data-visualrois/'\
    'sub-??/2ndlvl.gfeat/cope*.feat/stats/zstat1.nii.gz'

# contrast used by Sengupta et al. (2016) to create the PPA mask
VIS_VPN_COPES = OrderedDict({  # dicts are ordered from Python 3.7
    'sub-01': 'cope8',
    'sub-02': 'cope3',
    'sub-03': 'cope3',
    'sub-04': 'cope3',
    'sub-05': 'cope3',
    'sub-06': 'cope3',
    'sub-09': 'cope3',
    'sub-14': 'cope3',
    'sub-15': 'cope3',
    'sub-16': 'cope3',
    'sub-17': 'cope3',
    'sub-18': 'cope8',
    'sub-19': 'cope3',
    'sub-20': 'cope3'
})


def parse_arguments():
    '''
    '''
    parser = argparse.ArgumentParser(
        description='Creates mosaic of individual Bland-Altman-Plots'
    )

    parser.add_argument('-i',
                        default='./',
                        help='input directory')


    parser.add_argument('-o',
                        default='test',
                        help='output directory')

    args = parser.parse_args()

    inDir = args.i
    outDir = args.o

    return inDir, outDir


def find_files(pattern):
    '''
    '''
    found_files = glob(pattern)
    found_files = sort_nicely(found_files)

    return found_files


def sort_nicely(l):
    '''Sorts a given list in the way that humans expect
    '''
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    l.sort(key=alphanum_key)

    return l


def compute_means(data1, data2, log='n'):
    '''
    '''
    if len(data1) != len(data2):
        raise ValueError('data1 does not have the same length as data2.')

    data1 = np.asarray(data1)
    data2 = np.asarray(data2)

    if log == 'n':
        means = np.mean([data1, data2], axis=0)
    elif log == 'y':
        # what ever computation
        pass

    return means


def compute_diffs(data1, data2, log='n'):
    '''
    '''
    if len(data1) != len(data2):
        raise ValueError('data1 does not have the same length as data2.')

    data1 = np.asarray(data1)
    data2 = np.asarray(data2)

    if log == 'n':
        diffs = data1 - data2  # Difference between data1 and data2
    elif log == 'y':
        # what ever computation
        pass

    return diffs


def process_df(subj, zmaps_df, out_path, prediction):
    '''
    '''
    # get the contrasts' names from the column names
    zmap1 = zmaps_df.iloc[:, 1]
    zmap2 = zmaps_df.iloc[:, 0]

    # mask all voxels not contained in the PPA group mask
    ppa_grp_masked1 = zmap1.as_matrix()[zmaps_df['ppa_grp'].as_matrix() > 0]
    ppa_grp_masked2 = zmap2.as_matrix()[zmaps_df['ppa_grp'].as_matrix() > 0]

    # mask all voxels not contained in the individual PPA mask
    ppa_ind_masked1 = zmap1.as_matrix()[zmaps_df['ppa_ind'].as_matrix() > 0]
    ppa_ind_masked2 = zmap2.as_matrix()[zmaps_df['ppa_ind'].as_matrix() > 0]

#     ao_ind_masked1 = zmap1.as_matrix()[zmaps_df['ao_ind'].as_matrix() > 0]
#     ao_ind_masked2 = zmap2.as_matrix()[zmaps_df['ao_ind'].as_matrix() > 0]

    datasets = [
        [zmap1, zmap2],
        [ppa_grp_masked1, ppa_grp_masked2],
        [ppa_ind_masked1, ppa_ind_masked2],
#         [ao_ind_masked1, ao_ind_masked2]
    ]

    means_list = [compute_means(data1, data2) for data1, data2 in datasets]
    diffs_list = [compute_diffs(data1, data2) for data1, data2 in datasets]

    # Set up the axes with gridspec
    fig = plt.figure(figsize=(6, 6))
    grid = plt.GridSpec(6, 6, hspace=0.0, wspace=0.0)

    # add three subplots
    ax_scatter = fig.add_subplot(grid[1:, :-1])
    ax_xhist = fig.add_subplot(grid[0:1, 0:-1],
                               yticklabels=[],
                               sharex=ax_scatter)
    ax_yhist = fig.add_subplot(grid[1:, -1],
                               xticklabels=[],
                               sharey=ax_scatter)

    ax_scatter.text(5.1, 5.8, subj, fontsize=16, fontweight='bold')

    # plot voxel within occipitotemporal cortex
    plot_blandaltman(ax_scatter,
                     means_list[0],
                     diffs_list[0],
                     alpha=0.6,
                     c='darkgrey',
                     s=2)

    # plot voxels within PPA group overlap
    plot_blandaltman(ax_scatter,
                     means_list[1],
                     diffs_list[1],
                     alpha=1,
                     c='royalblue',
                     s=2)

    # plot voxels within individual PPA ROI
    plot_blandaltman(ax_scatter,
                     means_list[2],
                     diffs_list[2],
                     alpha=1,
                     c='r',
                     s=2)

#     # plot voxels within (thresholded) individual AO zmap
#     plot_blandaltman(ax_scatter,
#                      means_list[3],
#                      diffs_list[3],
#                      alpha=0.5,
#                      c='y',
#                      s=2)

    plot_histogram(ax_xhist, ax_yhist,
                   means_list[0], diffs_list[0],
                   alpha=1,
                   color='darkgrey')

    plot_histogram(ax_xhist, ax_yhist,
                   means_list[1], diffs_list[1],
                   alpha=1,
                   color='royalblue')

    plot_histogram(ax_xhist, ax_yhist,
                   means_list[2], diffs_list[2],
                   alpha=1,
                   color='r')

#     try:
#         plot_histogram(ax_xhist, ax_yhist,
#                        means_list[3], diffs_list[3],
#                        alpha=1,
#                        color='y')
#
#     except ValueError:
#         print(subj, 'has no significant cluster in primary AO contrast')

    # save it
    suffix = 'from_' + prediction.split('.nii')[0]
    out_file = ('%s_bland-altman_%s.png' % (subj, suffix))
    out_fpath = os.path.join(out_path, out_file)

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    plt.savefig(out_fpath,
                bbox_inches='tight',
                dpi=80)
    plt.close()


def plot_blandaltman(ax, means, diffs, *args, **kwargs):
    '''
    '''
    if len(means) != len(diffs):
        raise ValueError('means do not have the same length as diffs.')

    # annotation
    # variable subj is still a global here
    if subj in ['sub-01', 'sub-04', 'sub-09', 'sub-16', 'sub-19']:
        ax.set_ylabel('Difference between 2 measures', fontsize=16)

    if subj in ['sub-19', 'sub-20']:
        ax.set_xlabel('Average of 2 measures', fontsize=16)

    # draw the scattergram
    ax.scatter(means, diffs, *args, **kwargs)

    # set the size of the tick labels
    ax.xaxis.set_tick_params(labelsize=16)
    ax.yaxis.set_tick_params(labelsize=16)

    # limit the range of data shown
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)

    # draw horizontal and vertical line at 0
    ax.axhline(0, color='k', linewidth=0.5, linestyle='--')
    ax.axvline(0, color='k', linewidth=0.5, linestyle='--')

    return None


def plot_histogram(x_hist, y_hist, means, diffs, *args, **kwargs):
    '''
    '''
    # basic preparation of the axes
    x_hist.xaxis.set_tick_params(bottom=True, labelbottom=False)
    y_hist.yaxis.set_tick_params(bottom=True, labelbottom=False)

    # show vertical/horizontal zero line
    x_hist.axvline(0, color='k', linewidth=0.5, linestyle='--')
    y_hist.axhline(0, color='k', linewidth=0.5, linestyle='--')

    # plot histogram -> take KDE plot, s. below
    # x_hist.hist(means, 50, normed=True, histtype='bar',
    #             orientation='vertical', **kwargs)
    # y_hist.hist(diffs, 50, normed=True, histtype='bar',
    #             orientation='horizontal', **kwargs)

    # x_hist.set_yscale('log')
    # y_hist.set_xscale('log')

    # plot KDE
    xvalues = np.arange(-5.1, 5.1, 0.1)
    kde_means = gaussian_kde(means)
    kde_diffs = gaussian_kde(diffs)

    # KDE subplot on the top
    x_hist.plot(xvalues, kde_means(xvalues), **kwargs)
    # x_hist.fill_between(xvalues, kde_means(xvalues), 0, **kwargs)
    x_hist.set_ylim(0.015, 0.8)
    x_hist.set_yticks([0.2, 0.4, 0.6])  # , 1])
    x_hist.set_yticklabels(['.2', '.4', '.6'])  # , '1'])

    x_hist.yaxis.set_tick_params(labelsize=16)

    # KDE subplot on the right
    y_hist.plot(kde_diffs(xvalues), xvalues, **kwargs)
    # y_hist.fill_between(xvalues, kde_diffs(xvalues), 0, **kwargs)
    y_hist.set_xlim(0.015, .8)
    y_hist.set_xticks([0.2, 0.4, 0.6])  # , 1])
    y_hist.set_xticklabels(['.2', '.4', '.6'])  # , '1'])

    y_hist.xaxis.set_tick_params(labelsize=16)

    return None


def create_mosaic(in_pattern, dims, out_fpath, dpi):
    '''
    http://www.imagemagick.org/Usage/montage/
    '''

    # dimensions in columns*rows
    subprocess.call(
        ['montage',
         '-density', str(dpi),
         in_pattern,
         '-geometry', '+1+1',
         '-tile', dims,
         out_fpath])


def load_mask(subj):
    '''
    '''
    # open the mask of cortices (at the moment it justs the union of
    # individual PPAs)
    grp_ppa_fpath = GRP_PPA_PTTRN.replace('sub-??', subj)
#    print('PPA GRP mask:\t', grp_ppa_fpath)

    # subject-specific field of view in audio-description
    ao_fov_mask = AO_FOV_MASK.replace('sub-??', subj)
#    print('ind. FoV mask:\t', ao_fov_mask)

    # load the masks
    grp_ppa_img = nib.load(grp_ppa_fpath)
    ao_fov_img = nib.load(ao_fov_mask)

    # (dilated) gray matter mask; see constat at script's top
    #gm_fpath = GM_MASK.replace('sub-??', subj)
    #gm_img = nib.load(gm_fpath)
    #final_mask_data = grp_ppa_img.get_fdata() * gm_img.get_fdata()

    final_mask_data = grp_ppa_img.get_fdata() * ao_fov_img.get_fdata()
    final_mask_img = nib.Nifti1Image(final_mask_data,
                                     grp_ppa_img.affine,
                                     header=grp_ppa_img.header)

    return final_mask_img


if __name__ == "__main__":
    # get command line argument
    inDir, outDir = parse_arguments()

    vis_fpathes = find_files(VIS_ZMAP_PATTERN)

    # get pathes & filenames of all available zmaps
    subjs_pattern = os.path.join(inDir, 'sub-??')
    subjs_pathes = find_files(subjs_pattern)
    subjs = [re.search(r'sub-..', string).group() for string in subjs_pathes]
    subjs = sorted(list(set(subjs)))

    for prediction in PREDICTIONS[:]:
        # loop over subjects
        for subj in subjs:
            print('\nProcessing', subj)

            # initialize a dataframe
            zmaps_df = pd.DataFrame()

            # load predicted z-map of current subject
            pred_fpath = os.path.join(inDir, subj, PREDICTED_PREFIX + prediction)
            pred_fpath = pred_fpath.replace('sub-??', subj)
            pred_data = fmri_dataset(pred_fpath).samples.T
            # put the array into the dataframe
            zmaps_df['predicted'] = np.ndarray.flatten(pred_data)

            # load the visual localizer's z-map
            # filter for current subject
            fpathes = [x for x in vis_fpathes if subj in x]
            # filter for the subject's correct cope
            vis_fpath = [x for x in fpathes if VIS_VPN_COPES[subj] in x][0]
            vis_data = fmri_dataset(vis_fpath).samples.T
            # put the array into the dataframe
            zmaps_df['vis'] = np.ndarray.flatten(vis_data)

            # load the GRP PPA mask (in subject space)
            # (GRP PPA will be masked with individual FoV)
            final_mask_img = load_mask(subj)
            final_mask_data = final_mask_img.get_fdata()
            # put the array into the dataframe
            zmaps_df['ppa_grp'] = np.ndarray.flatten(final_mask_data)

            # load the subject-specific PPA mask (Sengupta et al., 2016)
            ppa_fpathes = find_files(PPA_MASK_PATTERN.replace('sub-*', subj))
            ppaIndData = fmri_dataset(ppa_fpathes).samples.sum(axis=0)
            # masked it with individual FoV (just to be sure)
            ao_fov_fpath = find_files(AO_FOV_MASK.replace('sub-??', subj))
            ao_fov_data = fmri_dataset(ao_fov_fpath).samples

            ppaIndMasked = ppaIndData * ao_fov_data

            zmaps_df['ppa_ind'] = np.ndarray.flatten(ppaIndMasked)

            # let the function do the calculation and the plotting
            process_df(subj, zmaps_df, outDir, prediction)

        # create the mosaic
        suffix = 'from_' + prediction.split('.nii')[0]
        infile_pattern = os.path.join(outDir, ('sub-??_bland-altman_%s.png' % suffix))
        out_fpath = os.path.join(outDir, 'subjs_bland-altman_%s.png' % suffix)

        create_mosaic(infile_pattern, '3x5', out_fpath, 80)  # columns x rows

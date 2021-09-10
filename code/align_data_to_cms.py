#!/usr/bin/env python3
'''
created on Mon March 29th 2021
author: Christian Olaf Haeusler
'''

from collections import OrderedDict
from glob import glob
from nilearn.input_data import NiftiMasker, MultiNiftiMasker
from scipy import stats
import brainiak.funcalign.srm
import argparse
import nibabel as nib
import ipdb
import numpy as np
import os
import random
import re


# constants
IN_PATTERN = 'sub-??/sub-??_task_aomovie-avmovie_run-1-8_bold-filtered.npy'

VIS_ZMAP_PATTERN = 'inputs/studyforrest-data-visualrois/'\
    'sub-*/2ndlvl.gfeat/cope*.feat/stats/zstat1.nii.gz'

AO_ZMAP_PATTERN = 'inputs/studyforrest-ppa-analysis/'\
    'sub-*/2nd-lvl_audio-ppa-ind.gfeat/cope1.feat/stats/zstat1.nii.gz'

MASK_PTTRN = 'sub-??/masks/in_bold3Tp2/grp_PPA_bin.nii.gz'
GM_MASK = 'sub-??/masks/in_bold3Tp2/gm_bin_dil_fov.nii.gz'

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
        description='load a pickled SRM and do some plotting'
    )

    parser.add_argument('-sub',
                        required=False,
                        default='sub-02',
                        help='subject to leave out (e.g. "subj-01")')

    parser.add_argument('-indir',
                        required=False,
                        default='test',
                        help='output directory (e.g. "sub-01")')

    parser.add_argument('-nfeat',
                        required=False,
                        default='30',
                        help='number of features (shared responses)')

    parser.add_argument('-niter',
                        required=False,
                        default='20',
                        help='number of iterations')

    args = parser.parse_args()

    sub = args.sub
    indir = args.indir
    n_feat = int(args.nfeat)
    n_iter = int(args.niter)

    return sub, indir, n_feat, n_iter

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


def load_mask(subj):
    '''
    '''
    # open the mask of cortices (at the moment it justs the union of
    # individual PPAs) and mask that with the individual (dilated) gray matter
    # in the audio-descriptions FoV
    mask_fpath = MASK_PTTRN.replace('sub-??', subj)

    # DEBUG / TO DO
    # MERGE PPA and (e.g.) FFA here before masking with GM in Fov

    # (dilated) gray matter mask; see constat at script's top
    gm_mask = GM_MASK.replace('sub-??', subj)

    # mask the area with individual (dilated) gray matter in FoV
    # of audio-description
    area_img = nib.load(mask_fpath)
    gm_img = nib.load(gm_mask)

    final_mask_data = area_img.get_fdata() * gm_img.get_fdata()
    final_mask_img = nib.Nifti1Image(final_mask_data,
                                     area_img.affine,
                                     header=area_img.header)


    return final_mask_img


if __name__ == "__main__":
    # read command line arguments
    left_out_subj, in_dir, n_feat, n_iter = parse_arguments()

    # get the subjects for which data are available
    SUBJS_PATH_PATTERN = 'sub-??'
    subjs_pathes = find_files(SUBJS_PATH_PATTERN)
    subjs = [re.search(r'sub-..', string).group() for string in subjs_pathes]
    # some filtering
    subjs = sorted(list(set(subjs)))

    for left_out_subj in subjs:
        # remove the currently processed subject from the list
        other_subjs = [x for x in subjs if x != left_out_subj]

        # prepare loading the SRM
        srm_fpath = os.path.join(
            in_dir, f'{left_out_subj}_srm_feat{n_feat}-iter{n_iter}.npz'
        )

        # load the srm from file
        print('Loading', srm_fpath)
        srm = load_srm(srm_fpath)

        # get the z-maps for the subjects that were used to create the CMS
        zmap_fpathes = [
            (VIS_ZMAP_PATTERN.replace('sub-*', x[0]).replace('cope*', x[1]))
            for x in VIS_VPN_COPES.items()]

#         zmap_fpathes = [
#             (AO_ZMAP_PATTERN.replace('sub-*', x[0]).replace('cope*', x[1]))
#             for x in VIS_VPN_COPES.items()]

        masked_zmaps = []
        for other_subj, zmap_fpath in zip(other_subjs, zmap_fpathes):
            print(other_subj)

            mask_img = load_mask(other_subj)
            # create instance of NiftiMasker used to mask the 4D time-series
            # which will be loaded next
            nifti_masker = NiftiMasker(mask_img=mask_img)

            # load the subject's zmap of the PPA contrast
            print(zmap_fpath)
            zmap_img = nib.load(zmap_fpath)
            zmap_img_affine = zmap_img.affine
            zmap_img_header = zmap_img.header

            zmap_img_data = zmap_img.get_fdata()
            # print(zmap_img_data.shape)

            # mask the image and get the data as np.ndarray
            nifti_masker.fit(zmap_img)
            print('before transform:', zmap_img_data.shape)
            masked_data = nifti_masker.transform(zmap_img)
            print('after transform:', masked_data.shape)
            masked_data = np.transpose(masked_data)
            print(f'zmap shape: {masked_data.shape}')
            print(f'subjects weight matrix: {srm.w_[other_subjs.index(other_subj)].shape}')

            masked_zmaps.append(masked_data)

        # aligned zmap to shared space
        # k feautures x t time-points
        # (1 time-point cause it's a zmap no time-series)
        zmaps_in_cms = srm.transform(masked_zmaps)

        # get the mean of features x t time-points
        matrix = np.stack(zmaps_in_cms)
        zmaps_cms_mean = np.mean(matrix, axis=0)

        start = 0
        end = 451
        in_fpath = os.path.join(in_dir, f'{left_out_subj}_wmatrix_{start}-{end}.npy')
        wmatrix = np.load(in_fpath)

        predicted = np.matmul(wmatrix, zmaps_cms_mean)

        # get the mask to perform its inverse transform
        mask_img = load_mask(left_out_subj)

        # create instance of NiftiMasker used to mask the 4D time-series
        # which will be loaded next
        nifti_masker = NiftiMasker(mask_img=mask_img)

        # initialize image
        print(zmap_fpath)
        zmap_img = nib.load(zmap_fpath)
        zmap_img_affine = zmap_img.affine
        zmap_img_header = zmap_img.header

        zmap_img_data = zmap_img.get_fdata()
        # print(zmap_img_data.shape)

        # mask the image and get the data as np.ndarray
        nifti_masker.fit(zmap_img)

        # get the data back in shape of the volume
        predicted_data = np.transpose(predicted)
        predicted_img = nifti_masker.inverse_transform(predicted_data)

        # adjust the name of the output file according to the input:
        if 'studyforrest-data-visualrois' in zmap_fpathes[0]:
            out_fpath = os.path.join(in_dir, f'{left_out_subj}_predicted_VIS_PPA.nii.gz')
        elif 'studyforrest-ppa-analysis' in zmap_fpathes[0]:
            out_fpath = os.path.join(in_dir, f'{left_out_subj}_predicted_AO_PPA.nii.gz')

        # save it
        nib.save(predicted_img, out_fpath)


    ### Z-SCORING OF TRANSFORMED DATA???
#     from scipy import stats
#     for subject in range(len(other_subjs)):
#         shared_zmaps[subject] = stats.zscore(shared_zmaps[subject], axis=1, ddof=1)

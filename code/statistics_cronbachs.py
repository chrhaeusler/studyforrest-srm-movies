#!/usr/bin/env python3
'''
created on Mon March 29th 2021
author: Christian Olaf Haeusler
'''

from collections import OrderedDict
from glob import glob
from nilearn.input_data import NiftiMasker, MultiNiftiMasker
from os.path import join as opj
import argparse
import csv
import nibabel as nib
import numpy as np
import pandas as pd
import re

# constants
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

# pattern for the localizer contrast
VIS_FRST_LVL_PTTRN = 'inputs/studyforrest-data-visualrois/' \
    'sub-*/run-*.feat/stats/zstat?.nii.gz'

# pattern for the PPA localized via movie (primary contrast)
AV_FRST_LVL_PTTRN = 'inputs/studyforrest-ppa-analysis/' \
    'sub-*/run-*_movie-ppa-ind.feat/stats/zstat1.nii.gz'

# pattern for the PPA localized via audio-description (primary contrast)
AO_FRST_LVL_PTTRN = 'inputs/studyforrest-ppa-analysis/' \
    'sub-*/run-*_audio-ppa-ind.feat/stats/zstat1.nii.gz'

# pattern for the mask in subject space
GRP_PPA_PTTRN = 'sub-??/masks/in_bold3Tp2/grp_PPA_bin.nii.gz'
AO_FOV_MASK = 'sub-??/masks/in_bold3Tp2/audio_fov.nii.gz'



def parse_arguments():
    '''
    '''
    parser = argparse.ArgumentParser(
        description='load a pickled SRM and do some plotting'
    )

    parser.add_argument('-outDir',
                        required=False,
                        default='results',
                        help='the ouput directory')

    parser.add_argument('-vis',
                        required=False,
                        default='inputs/studyforrest-data-visualrois/' \
                        'sub-*/run-*.feat/stats/zstat?.nii.gz',
                        help='pattern of 1st lvl z-maps')

    parser.add_argument('-grpPPAmask',
                        required=False,
                        default='sub-??/masks/in_bold3Tp2/grp_PPA_bin.nii.gz',
                        help='pattern of PPA group mask in subject-space')

    parser.add_argument('-aoFOVmask',
                        required=False,
                        default='sub-??/masks/in_bold3Tp2/audio_fov.nii.gz',
                        help='pattern of subject-specific FOV mask')

    args = parser.parse_args()

    outDir = args.outDir
    # visPattern = args.vis
    groupPPAmask = args.grpPPAmask
    aoFOVmask = args.aoFOVmask

    return outDir, groupPPAmask, aoFOVmask


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


def load_mask(subj, maskPattern, aoFOVpattern):
    '''
    '''
    # open the mask of cortices (at the moment it justs the union of
    # individual PPAs)
    grp_ppa_fpath = maskPattern.replace('sub-??', subj)
#    print('PPA GRP mask:\t', grp_ppa_fpath)

    # subject-specific field of view in audio-description
    ao_fov_mask = aoFOVpattern.replace('sub-??', subj)
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


def cronbach_alpha(itemscores):
    '''Calculates Cronbach's alpha for a given matrix
    '''
    itemscores = np.asarray(itemscores)
    itemvars = itemscores.var(axis=1, ddof=1)
    tscores = itemscores.sum(axis=0)
    nitems = len(itemscores)

    return nitems / (nitems-1.) * (1 - itemvars.sum() / tscores.var(ddof=1))


def cronbach_alpha_alt(matrix):
    '''alternative calculation of Cronbach's alpha for a given matrix

    Args:
        matrix: ndarray. shape 'voxel x run'

    Returns:
        alpha: float
    '''
    numItems = matrix.shape[1]
    # it's not necessary to specify ddof, as the term appears in the
    # denominator and numerator, and cancels.
    sumItemVar = matrix.var(axis=0).sum()
    varSumOfItems = matrix.sum(axis=1).var()
    alpha = numItems / (numItems - 1) * (1 - sumItemVar / varSumOfItems)

    return alpha


if __name__ == "__main__":
    # read command line arguments
    outDir, groupPPApattern, aoFOVpattern = parse_arguments()

    # get the subjects for which data are available
    subjsPathPattern = 'sub-??'
    subjsPathes = find_files(subjsPathPattern)
    subjs = [re.search(r'sub-..', string).group() for string in subjsPathes]
    subjs = sorted(list(set(subjs)))

    criterions = ['visual localizer',
                  'movie',
                  'audio-description']

    toWrite = [["sub", 'stimulus', "Cronbach's a", 'number of voxels']]

    # loop over PPAs localized via VIS, AV, and AO
    for criterion in criterions:
        print(f'\n{criterion}')

        # create the list of all subjects' z-maps
        if criterion == 'visual localizer':
            # use the correct cope that was used in Sengupta et al.
            frstLvlPttrn = [
                (VIS_FRST_LVL_PTTRN.replace('sub-*', x[0]).replace('?', x[1][-1]))
                for x in VIS_VPN_COPES.items()]
        elif criterion == 'movie':
            frstLvlPttrn = [AV_FRST_LVL_PTTRN.replace('sub-*', x) for x in subjs]
        elif criterion == 'audio-description':
            frstLvlPttrn = [AO_FRST_LVL_PTTRN.replace('sub-*', x) for x in subjs]

        # loop over the subjects
        resultsForCriterion = []
        for subj in subjs[:]:
            # filter input patterns of first run for current subj
            zmapPattern = [x for x in frstLvlPttrn if subj in x][0]
            zmapFpathes = find_files(zmapPattern)

            # load the mask (combines PPA + gray matter)
            mask_img = load_mask(subj, groupPPApattern, aoFOVpattern)
            # create instance of NiftiMasker used to mask the 4D time-series
            nifti_masker = NiftiMasker(mask_img=mask_img)

            # loops through the runs
            fourRunsList = []
            for zmapFpath in zmapFpathes:
                # print(zmapFpath)
                # load the current's run image
                zmap_img = nib.load(zmapFpath)

                # mask the VIS z-map
                nifti_masker.fit(zmap_img)
                masked_data = nifti_masker.transform(zmap_img)
                fourRunsList.append(masked_data)

            fourRuns = np.concatenate(fourRunsList, axis=0)
            cronbach = cronbach_alpha(fourRuns)
            print(f'{subj}, {masked_data.shape[1]} voxel, a={round(cronbach, 2)}')
            resultsForCriterion.append([subj, criterion, cronbach, masked_data.shape[1]])

        # print results to screen
        alphasForPrint = [str(round(x[2], 2)) for x in resultsForCriterion[1:]]
        print(', '.join(alphasForPrint))
        # print descriptive statistics, too
        toSummarize = pd.DataFrame(resultsForCriterion)
        print(toSummarize[2].describe())

        # add current paradigm's results
        toWrite.extend(resultsForCriterion)

    with open(opj(outDir, 'statistics_cronbachs.csv'), 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerows(toWrite)

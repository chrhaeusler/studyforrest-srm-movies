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
import pandas as pd
import re
import subprocess


# constants
IN_PATTERN = 'sub-??/sub-??_task_aomovie-avmovie_run-1-8_bold-filtered.npy'

# input directory of templates & transforms
TNT_DIR = 'inputs/studyforrest-data-templatetransforms'
# group BOLD image (reference image)
XFM_REF = os.path.join(TNT_DIR,
                       'templates/grpbold3Tp2/',
                        'brain.nii.gz')
# pre-transform (affine matrix)
XFM_MAT = os.path.join(TNT_DIR,
                       'templates/grpbold3Tp2/xfm/',
                       'mni2tmpl_12dof.mat'
                       )

GRP_PPA_PTTRN = 'sub-??/masks/in_bold3Tp2/grp_PPA_bin.nii.gz'
AO_FOV_MASK = 'sub-??/masks/in_bold3Tp2/audio_fov.nii.gz'
GM_MASK = 'sub-??/masks/in_bold3Tp2/gm_bin_dil_fov.nii.gz'

AO_ZMAP_PATTERN = 'inputs/studyforrest-ppa-analysis/'\
    'sub-*/2nd-lvl_audio-ppa-ind.gfeat/cope1.feat/stats/zstat1.nii.gz'

VIS_ZMAP_PATTERN = 'inputs/studyforrest-data-visualrois/'\
    'sub-*/2ndlvl.gfeat/cope*.feat/stats/zstat1.nii.gz'

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


# the models that we have
models = [
    'srm-ao-av',
    'srm-ao-av-vis'
]

# and vary the amount of TRs used for alignment
starts_ends = [
    (0, 451,  'AO', 1),     # AO, 1 run
    (0, 892,  'AO', 2),     # AO, 2 runs
    (0, 1330, 'AO', 3),     # AO, 3 runs
    (0, 1818, 'AO', 4),     # AO, 4 runs
    (0, 2280, 'AO', 5),     # AO, 5 runs
    (0, 2719, 'AO', 6),     # AO, 6 runs
    (0, 3261, 'AO', 7),     # AO, 7 runs
    (0, 3524, 'AO', 8),     # AO, 8 runs
    # (0,    7123, 16),  # AO & AV
    (3524, 3975, 'AV', 1),  # AV, 1 run
    (3524, 4416, 'AV', 2),  # AV, 2 runs
    (3524, 4854, 'AV', 3),  # AV, 3 runs
    (3524, 5342, 'AV', 4),  # AV, 4 runs
    (3524, 5804, 'AV', 5),  # AV, 5 runs
    (3524, 6243, 'AV', 6),  # AV, 6 runs
    (3524, 6785, 'AV', 7),  # AV, 7 runs
    (3524, 7123, 'AV', 8)  # AV, 8 runs
]


def parse_arguments():
    '''
    '''
    parser = argparse.ArgumentParser(
        description='load a pickled SRM and do some plotting'
    )

    parser.add_argument('-indir',
                        required=False,
                        default='test',
                        help='output directory (e.g. "sub-01")')

    parser.add_argument('-model',
                        required=False,
                        default='srm-ao-av-vis',
                        help='the fitted model')

    parser.add_argument('-nfeat',
                        required=False,
                        default='10',
                        help='number of features (shared responses)')

    parser.add_argument('-niter',
                        required=False,
                        default='30',
                        help='number of iterations')

    args = parser.parse_args()

    indir = args.indir
    model = args.model
    n_feat = int(args.nfeat)
    n_iter = int(args.niter)

    return indir, model, n_feat, n_iter


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


def transform_ind_ppas(zmap_fpathes, subjs):
    '''
    '''
    for source_subj in subjs:
        # filter pathes for the current subject
        zmap_fpath = [x for x in zmap_fpathes if source_subj in x][0]
        # name of output fule
        out_path = (os.path.join(in_dir, 'masks', 'in_mni'))

        if 'studyforrest-data-visualrois' in zmap_fpath:
            out_file = f'{source_subj}_VIS-PPA.nii.gz'
            out_fpath = os.path.join(out_path, out_file)
        elif '2nd-lvl_audio-ppa-ind' in zmap_fpath:
            out_file = f'{source_subj}_AO-PPA.nii.gz'
            out_fpath = os.path.join(out_path, out_file)
        else:
            print('unkown source for PPA (must be from VIS or AO)')
            continue

        # create the output path
        os.makedirs(out_path, exist_ok=True)

        print(f'{source_subj}: from bold3Tp to MNI using {zmap_fpath}')
        #
        subj2templWarp = os.path.join(TNT_DIR,
                                    source_subj,
                                    'bold3Tp2/in_grpbold3Tp2/'
                                    'subj2tmpl_warp.nii.gz'
                                    )

        # warp the subjec-specific VIS PPA to MNI space
        warp_subj_to_mni(zmap_fpath, out_fpath,
                        subj2templWarp, XFM_REF)

        # warp from MNI to every other subject space
        ppa_in_mni_fpath = out_fpath
        print(f'{source_subj}: from MNI to other subjs using {ppa_in_mni_fpath}')
        for target_subj in subjs:
            # do not transform the current's subject volume back
            # into its own bold3Tp2
            if target_subj != source_subj:
                # create the output path & filename
                out_path = os.path.join(in_dir,
                                        target_subj,
                                        'masks',
                                        'in_bold3Tp2'
                                        )
                os.makedirs(out_path, exist_ok=True)
                out_file = os.path.basename(ppa_in_mni_fpath)
                out_fpath = os.path.join(out_path, out_file)

                # the path of the (individual) reference image
                subj_ref = os.path.join(TNT_DIR,
                                        target_subj,
                                        'bold3Tp2/brain.nii.gz')

                # the volume providing warp/coefficient
                subj_warp = os.path.join(TNT_DIR,
                                        target_subj,
                                        'bold3Tp2/in_grpbold3Tp2/'
                                        'tmpl2subj_warp.nii.gz'
                                        )

                # do the warping from MNI to bold3Tp2 by calling the function
                print('warp to', out_fpath)
                warp_mni_to_subj(
                    ppa_in_mni_fpath,
                    out_fpath,
                    subj_ref,
                    subj_warp
                )

    return None


def warp_subj_to_mni(inputVol, outputVol, indWarp, xfmRef):
    '''
    '''
    # warp only in case the output file does not already exists:
    # if not os.path.exists(outputVol):
        # make sure inputs are locally available
    if not os.path.exists(indWarp):
        subprocess.call(['datalad', 'get', indWarp])

    if not os.path.exists(xfmRef):
        subprocess.call(['datalad', 'get', xfmRef])

    # call FSL's applywarp
    subprocess.call(
        ['applywarp',
            '-i', inputVol,
            '-o', outputVol,
            '-r', xfmRef,
            '-w', indWarp,
            '--interp=nn'
            # '--premat=premat'
            ]
    )

    return None


def warp_mni_to_subj(inputVol, outputVol, ind_ref, ind_warp):
    '''
    '''
    # warp only in case the output file does not already exists:
    # if not os.path.exists(outputVol):
        # make sure inputs are locally available
    if not os.path.exists(inputVol):
        subprocess.call(['datalad', 'get', inputVol])

    if not os.path.exists(ind_ref):
        subprocess.call(['datalad', 'get', ind_ref])

    if not os.path.exists(ind_warp):
        subprocess.call(['datalad', 'get', ind_warp])

    subprocess.call(
        ['applywarp',
         '-i', inputVol,
         '-o', outputVol,
         '-r', ind_ref,
         '-w', ind_warp,
         '--interp=nn'
         ]
    )

    return None


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


def transform_in_n_out(masked_zmaps, srm, wmatrix):
    '''
    '''
    # a) solution without brainIAK
    zmaps_in_ind = []
    for zmap, left_out_wmatrix in zip(masked_zmaps, srm.w_):
        intermediate_matrix = wmatrix.dot(left_out_wmatrix.T)
        zmap_in_ind = intermediate_matrix.dot(zmap)
        zmap_in_ind = np.transpose(zmap_in_ind)
        zmaps_in_ind.append(zmap_in_ind)

    # b) solution with brainIAK
    # aligned zmaps to shared space
    # k feautures x t time-points
    # (1 time-point cause it's a zmap no time-series)
    zmaps_in_cms = srm.transform(masked_zmaps)

#     ### THIS IS TAKING THE MEAN OF 'zmaps' aligned in CMS
#     # get the mean of features x t time-points
#     matrix = np.stack(zmaps_in_cms)
#     zmaps_cms_mean = np.mean(matrix, axis=0)
#
#     # transform from CMS into vol
#     predicted = np.matmul(wmatrix, zmaps_cms_mean)
#     predicted_data = np.transpose(predicted)

    # transform every zmap into left-out subjects space first
    # then take the mean
    zmaps_in_ind = []
    for zmap_in_cms in zmaps_in_cms:
        zmap_in_left_out = np.matmul(wmatrix, zmap_in_cms)
        zmap_in_left_out = np.transpose(zmap_in_left_out)
        zmaps_in_ind.append(zmap_in_left_out)

    # take the mean
    matrix = np.stack(zmaps_in_ind)
    predicted_data = np.mean(matrix, axis=0)

    return predicted_data


def predict_from_cms(left_out_subj, subjs, zmap_fpathes, start, end):
    '''
    '''
    # get a list of non-left-out subjects
    other_subjs = [x for x in subjs if x != left_out_subj]

    # load the left-out subject's transformation matrix
    # that will be used to tranform the data from CMS into brain volume
    in_fpath = os.path.join(in_dir, left_out_subj,
                            f'wmatrix_{model}_feat{n_feat}_{start}-{end}.npy')
    wmatrix = np.load(in_fpath)

    # load the SRM (based on non-left-out subjects' data)
    srm_fpath = os.path.join(in_dir,
                             left_out_subj,
                             f'{model}_feat{n_feat}-iter{n_iter}.npz')
    # load it
    srm = load_srm(srm_fpath)

    # load the non-left-out subjects' empirical z-maps, and the mask
    # loop over the subjects
    masked_zmaps = []
    for other_subj in other_subjs:
        # filter for the z-map of the current non-left-out subject
        zmap_fpath = [x for x in zmap_fpathes if other_subj in x][0]

        # load the subject's zmap of the PPA contrast
#        print(f'\nz-map for {other_subj}: {zmap_fpath}')
        zmap_img = nib.load(zmap_fpath)

        # load the mask for current subject
#        print('masking empirical data of non-left-out subject')
        mask_img = load_mask(other_subj)
        # create instance of NiftiMasker
        other_subj_masker = NiftiMasker(mask_img=mask_img)

        # mask the image and get the data as np.ndarray
        other_subj_masker.fit(zmap_img)
        masked_data = other_subj_masker.transform(zmap_img)
        masked_data = np.transpose(masked_data)
#         print(f'shape of z-map (transposed): {masked_data.shape}')
#         print(f'shape of weight matrix: {srm.w_[other_subjs.index(other_subj)].shape}')
        masked_zmaps.append(masked_data)

    # transform the data into the target subject's space
    predicted_data = transform_in_n_out(masked_zmaps, srm, wmatrix)

    # get the mask of the left-out subject to perform its inverse transform
#    print('loading mask to perform inverse transform')
    mask_img = load_mask(left_out_subj)
    # create instance of NiftiMasker
    nifti_masker = NiftiMasker(mask_img=mask_img)
    # fit the masker
    nifti_masker.fit(zmap_img)
    # transform the predicted data from array to volume
    predicted_img = nifti_masker.inverse_transform(predicted_data)

    # adjust the name of the output file according to the input:
    if 'studyforrest-data-visualrois' in zmap_fpathes[0]:
        out_fpath = os.path.join(
            in_dir, left_out_subj,
            f'predicted-VIS-PPA_from_{model}_feat{n_feat}_{start}-{end}.nii.gz')
    elif '2nd-lvl_audio-ppa-ind' in zmap_fpathes[0]:
        out_fpath = os.path.join(
            in_dir, left_out_subj,
            f'predicted-AO-PPA_from_{model}_feat{n_feat}_{start}-{end}.nii.gz')

    # save it
    nib.save(predicted_img, out_fpath)

    return predicted_data


def predict_from_ana(left_out_subj, nifti_masker, subjs, zmap_fpathes):
    '''
    '''
    # filter out the left out subj
    other_subjs = [x for x in subjs if x != left_out_subj]

    ppa_arrays = []
    for other_subj in other_subjs:
        # open the other subject's PPA z-map which was transformed in the
        # subject-specific space of the current subject
        if 'studyforrest-data-visualrois' in zmap_fpathes[0]:
            which_PPA = 'VIS'
        elif '2nd-lvl_audio-ppa-ind' in zmap_fpathes[0]:
            which_PPA = 'AO'
        else:
            print('unkown source for PPA (must be from VIS or AO)')
            break

        ppa_img_fpath = os.path.join(
            in_dir, left_out_subj,
            'masks', 'in_bold3Tp2',
            f'{other_subj}_{which_PPA}-PPA.nii.gz')

        ppa_img = nib.load(ppa_img_fpath)
        masked_array = nifti_masker.transform(ppa_img)
        masked_array = np.transpose(masked_array)

        # append the current subject's array to the list of arrays
        ppa_arrays.append(masked_array)

    stacked_arrays = np.stack(ppa_arrays)
    mean_anat_arr = np.mean(stacked_arrays, axis=0)
    mean_anat_arr = np.transpose(mean_anat_arr)

    # transform the predicted data from array to volume
    mean_anat_img = nifti_masker.inverse_transform(mean_anat_arr)

    out_fpath = os.path.join(in_dir,
                             left_out_subj,
                             f'predicted-{which_PPA}-PPA_from_anatomy.nii.gz')

    nib.save(mean_anat_img, out_fpath)

    return mean_anat_arr


def run_the_predictions(zmap_fpathes, subjs):
    '''
    '''
    # just take the second model that includes VIS
    for model in models[1:]:
        print(f'model:\t{model}_feat{n_feat}_iter{n_iter}.npz')

        # the list for the dataframe that will build from the list
        for_dataframe = []

        # compute the anatomical prediction for every subject first
        empirical_arrays = []
        anat_pred_arrays = []

        for left_out_subj in subjs[:]:
            # load the subject's zmap of the PPA contrast
            zmap_fpath = [x for x in zmap_fpathes if left_out_subj in x][0]
            zmap_img = nib.load(zmap_fpath)

            # load the mask (combines PPA + gray matter)
            mask_img = load_mask(left_out_subj)
            # create instance of NiftiMasker used to mask the 4D time-series
            nifti_masker = NiftiMasker(mask_img=mask_img)
            nifti_masker.fit(zmap_img)

            # mask the VIS z-map
            masked_data = nifti_masker.transform(zmap_img)
            masked_data = np.transpose(masked_data)
            masked_data = masked_data.flatten()
            empirical_arrays.append(masked_data)

            # predict from anatomy
            anat_pred_array = predict_from_ana(left_out_subj,
                                               nifti_masker,
                                               subjs,
                                               zmap_fpathes)
            # append to the list of arrays
            anat_pred_arrays.append(anat_pred_array.flatten())

        # compute the correlations of empirical vs. prediction from anatomy
        emp_vs_anat = [stats.pearsonr(empirical_arrays[idx],
                                      anat_pred_arrays[idx]) for idx
                                      in range(len(empirical_arrays))]

        # compute the functional prediction for increasing number of
        # fMRI runs used
        for start, end, stim, runs in starts_ends[:]:
            print(f'\nTRs:\t{start}-{end}')

            # following list will hold, for every subject, an array
            # of z-values
            func_pred_arrays = []

            # for the current number of runs used, loop through the subjects
            for left_out_subj in subjs[:]:
                # predict from CMS
                func_pred_array = predict_from_cms(left_out_subj,
                                                   subjs,
                                                   zmap_fpathes,
                                                   start,
                                                   end)
                # append to the list of arrays
                func_pred_arrays.append(func_pred_array.flatten())

            # compute the correlations of empirical vs. prediction from
            # functional alignment (with currently used no. of runs)
            emp_vs_func = [stats.pearsonr(empirical_arrays[idx],
                                          func_pred_arrays[idx]) for idx
                                          in range(len(empirical_arrays))]

            # print the result of the currently used runs / stimulus
            print('subject\temp. vs. anat\temp. vs. cms')

            for idx, subj in enumerate(subjs):
                print(f'{subj}\t{round(emp_vs_anat[idx][0], 2)}\t{round(emp_vs_func[idx][0], 2)}')

            # get the data for the currently used TRs for aligment in shape
            # so they can later be stored in the dataframe
            if stim == 'AO':
                predictor = 'audio-description'
            elif stim == 'AV':
                predictor = 'movie'

            func_lines = [[subj, predictor, runs, corr[0]] for subj, corr in zip(subjs, emp_vs_func)]
            # list of line for the dataframe
            for_dataframe.extend(func_lines)

        # add the correlations of prediction from anatomy vs. empirical data
        anat_lines = [[subj, 'anatomy', 0, corr[0]]
                      for subj, corr in zip(subjs, emp_vs_anat)]

        # put the correlations per subject into the dataframe
        for_dataframe.extend(anat_lines)

        # prepare the dataframe for the current model
        df = pd.DataFrame(for_dataframe, columns = ['sub',
                                                    'prediction from',
                                                    'runs',
                                                    'Pearson\'s r'])

        # adjust the name of the output file according to the input:
        if 'studyforrest-data-visualrois' in zmap_fpathes[0]:
            which_PPA = 'VIS'

        elif '2nd-lvl_audio-ppa-ind' in zmap_fpathes[0]:
            which_PPA = 'AO'

        # save the dataframe for the currently used CMS
        df.to_csv(f'{in_dir}/corr-emp{which_PPA}-vs-func.csv', index=False)

    return None


if __name__ == "__main__":
    # read command line arguments
    in_dir, model, n_feat, n_iter = parse_arguments()

    # get the subjects for which data are available
    subjs_path_pattern = 'sub-??'
    subjs_pathes = find_files(subjs_path_pattern)
    subjs = [re.search(r'sub-..', string).group() for string in subjs_pathes]
    subjs = sorted(list(set(subjs)))

    # for later prediction from anatomy, we need to transform the
    # subject-specific z-maps from the localizer into MNI space
    # (and later transform then into the subject-space of the left-out subject

    ### VIS LOCALIZER PPA
    # create the list of all subjects' VIS zmaps by substituting the
    # subject's string and the correct cope that was used in Sengupta et al.
    zmap_fpathes = [
        (VIS_ZMAP_PATTERN.replace('sub-*', x[0]).replace('cope*', x[1]))
        for x in VIS_VPN_COPES.items()]

    print('\nTransforming VIS z-maps into MNI and into other subjects\' space')
    transform_ind_ppas(zmap_fpathes, subjs)

    run_the_predictions(zmap_fpathes, subjs)

    ### AO STIMULUS PPA
    zmap_fpathes = [
        (AO_ZMAP_PATTERN.replace('sub-*', x[0]))
        for x in VIS_VPN_COPES.items()]

    print('\nTransforming AO z-maps into MNI and into other subjects\' space')
    transform_ind_ppas(zmap_fpathes, subjs)

    run_the_predictions(zmap_fpathes, subjs)

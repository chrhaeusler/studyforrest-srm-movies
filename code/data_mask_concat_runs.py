#!/usr/bin/env python3
'''
created on Mon March 29th 2021
author: Christian Olaf Haeusler
'''

from glob import glob
from nilearn.input_data import NiftiMasker, MultiNiftiMasker
import argparse
import nibabel as nib
import numpy as np
import os
import re


# constants
MASK_PTTRN = 'sub-??/masks/in_bold3Tp2/grp_PPA_bin.nii.gz'
GM_MASK = 'sub-??/masks/in_bold3Tp2/gm_bin_dil_fov.nii.gz'
IN_FILE_PTTRN = 'sub-??/sub-??_task-a?movie_run-?_bold_filtered.nii.gz'
VIS_FILE_PTTRN = 'sub-??/sub-??_task-visloc_run-?_bold_filtered.nii.gz'

def parse_arguments():
    '''
    '''
    parser = argparse.ArgumentParser(
        description='masks and concatenates runs of AO & AV stimulus'
    )

    parser.add_argument('-sub',
                        required=False,
                        default='sub-01',
                        help='subject to process (e.g. "subj-01")')

    parser.add_argument('-outdir',
                        required=False,
                        default='test',
                        help='output directory (e.g. "sub-01")')

    args = parser.parse_args()

    sub = args.sub
    outdir = args.outdir

    return sub, outdir


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
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    l.sort(key=alphanum_key)

    return l


if __name__ == "__main__":
    # read command line arguments
    subj, out_dir = parse_arguments()
    subj = subj.strip('/')

    print('\nProcessing', subj)

    in_pattern = IN_FILE_PTTRN.replace('sub-??', subj)
    in_fpathes = find_files(in_pattern)

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

    # create instance of NiftiMasker used to mask the 4D time-series
    # which will be loaded next
    nifti_masker = NiftiMasker(mask_img=final_mask_img)

    # start with the first run of the AO stimulus
    print('Processing data of audio-description & movie')
    print(in_fpathes[0])
    first_img = nib.load(in_fpathes[0])
    first_img_affine = first_img.affine
    first_img_header = first_img.header
    # all_imgs_data = all_imgs.get_fdata()
    # print(all_imgs_data.shape)

    # mask the 4D image and get the data as np.ndarray
    masked_data = nifti_masker.fit_transform(first_img)
    masked_data = np.transpose(masked_data)
    print(masked_data.shape)

    # concatenate remaining runs from AO & AV
    # nibabel.funcs.concat_images(images, check_affines=True, axis=None)
    # which does not allows setting a mask)
    for run, in_fpath in enumerate(in_fpathes[1:]):  # first index is 'first_img'
        run = run + 2
        # DEBUGGING / CHECK
        print(in_fpath)
        # load image
        new_img = nib.load(in_fpath)
        # mask the image and get the data as np.ndarray
        # nifti_masker = NiftiMasker(mask_img=mask_fpath)
        # nifti_masker ist the same as above anyway
        masked_new_data = nifti_masker.fit_transform(new_img)  # returns numpy.ndarray
        masked_new_data = np.transpose(masked_new_data)

        # concatenate current time-series to previous time-series
        masked_data = np.concatenate(
            (masked_data, masked_new_data), axis=1
        )

        # print current size of the now extended time-series
        print(masked_data.shape)

    out_file = f'{subj}_task_aomovie-avmovie_run-1-8_bold-filtered.npy'
    out_fpath = os.path.join(out_dir, out_file)

    # only needed is path "test" is used
    os.makedirs(os.path.dirname(out_fpath), exist_ok=True)

    np.save(out_fpath, masked_data)

    # VISUAL LOCALIZER
    # do the same for the runs of the visual localizer
    vis_pattern = VIS_FILE_PTTRN.replace('sub-??', subj)
    vis_fpathes = find_files(vis_pattern)

    # start with the first run of the visual localizer
    print('Processing data of visual localizer')
    print(vis_fpathes[0])
    first_img = nib.load(vis_fpathes[0])
    first_img_affine = first_img.affine
    first_img_header = first_img.header
    # all_imgs_data = all_imgs.get_fdata()
    # print(all_imgs_data.shape)

    # mask the 4D image and get the data as np.ndarray
    masked_data = nifti_masker.fit_transform(first_img)
    masked_data = np.transpose(masked_data)
    print(masked_data.shape)

    # concatenate remaining runs from visual localizer
    for run, vis_fpath in enumerate(vis_fpathes[1:]):  # first index is 'first_img'
        run = run + 2
        # DEBUGGING / CHECK
        print(vis_fpath)
        # load image
        new_img = nib.load(vis_fpath)
        # mask the image and get the data as np.ndarray
        # nifti_masker = NiftiMasker(mask_img=mask_fpath)
        # nifti_masker is the same as above anyway
        masked_new_data = nifti_masker.fit_transform(new_img)  # returns numpy.ndarray
        masked_new_data = np.transpose(masked_new_data)

        # concatenate current time-series to previous time-series
        masked_data = np.concatenate(
            (masked_data, masked_new_data), axis=1
        )

        # print current size of the now extended time-series
        print(masked_data.shape)

    out_file = f'{subj}_task_visloc_run-1-4_bold-filtered.npy'
    out_fpath = os.path.join(out_dir, out_file)

    # only needed if path "test" is used
    os.makedirs(os.path.dirname(out_fpath), exist_ok=True)

    np.save(out_fpath, masked_data)

    # save as Nifti1Image
#     all_data_img = nib.Nifti1Image(all_imgs_data,
#                                     all_imgs_affine,
#                                     header=all_imgs_header)
#     nib.save(all_data_img, out_fpath.replace('.npy', '.nii.gz')

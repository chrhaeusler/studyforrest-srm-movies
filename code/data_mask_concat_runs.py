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
IN_FILE_PTTRN = 'sub-??/sub-??_task-a?movie_run-?_bold_filtered.nii.gz'


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

    print('Processing', subj)

    in_fpathes = find_files(IN_FILE_PTTRN.replace('sub-??', subj))


    ##### READ the mask and give as an read-in image and not as path
    ##### will allow to do some merging of masks before
    mask_fpath = MASK_PTTRN.replace('sub-??', subj)

    # create instance of NiftiMasker used to mask the 4D time-series
    # which will be loaded next
    nifti_masker = NiftiMasker(mask_img=mask_fpath)

    # initialize image
    print(in_fpathes[0])
    first_img = nib.load(in_fpathes[0])
    first_img_affine = first_img.affine
    first_img_header = first_img.header
    # all_imgs_data = all_imgs.get_fdata()
    # print(all_imgs_data.shape)

    # mask the image and get the data as np.ndarray
    masked_data = nifti_masker.fit_transform(first_img)
    masked_data = np.transpose(masked_data)
    print(masked_data.shape)
    # remove first in_fpath from list

    # concatenate images
    # nibabel.funcs.concat_images(images, check_affines=True, axis=None)
    # which does not allows setting a mask)
    for run, in_fpath in enumerate(in_fpathes[1:]):  # first index is 'first_img'
        run = run + 2
        # DEBUGGING / CHECK
        print('\n' + in_fpath)
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

        print(masked_data.shape)

    out_fpath = os.path.join(
        out_dir,
        f'{subj}_task_aomovie-avmovie_run-1-8_bold-filtered.npy')

    # only needed for testpath
    os.makedirs(os.path.dirname(out_fpath), exist_ok=True)

    np.save(out_fpath, masked_data)

    # save as Nifti1Image
#     all_data_img = nib.Nifti1Image(all_imgs_data,
#                                     all_imgs_affine,
#                                     header=all_imgs_header)
#     nib.save(all_data_img, out_fpath.replace('.npy', '.nii.gz')

    print("\nEnd of Script")

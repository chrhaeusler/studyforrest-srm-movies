#!/usr/bin/env python3
'''
author: Christian Olaf Häusler
created on Wednesday, 31 March 2021
'''

from glob import glob
import subprocess
import nibabel as nib
import numpy as np
import os
import re
import scipy.ndimage


# constants
# path of the subdataset providing templates and transformatiom matrices
TNT_DIR = 'inputs/studyforrest-data-templatetransforms'

# the path that contains mask (input and output purpose)
ROIS_PATH = 'masks'

# the path to check for which subjects we have (filtered) functional data
# that were used to localize the PPA via movie and audio-description
SUBJS_PATH_PATTERN = 'inputs/studyforrest-ppa-analysis/sub-??/' + \
    'run-?_*-ppa-ind.feat'


def find_files(pattern):
    '''
    '''
    def sort_nicely(l):
        '''Sorts a given list in the way that humans expect
        '''
        convert = lambda text: int(text) if text.isdigit() else text
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        l.sort(key=alphanum_key)

        return l

    found_files = glob(pattern)
    found_files = sort_nicely(found_files)

    return found_files


def create_audio_FoV_mask(subj, filt4d_fpath, out_fpath):
    '''creates a binarized 3D image taken from a 4D time-series
    '''
    if not os.path.exists(out_fpath):
        # download data via datalad if needed
        if not os.path.exists(filt4d_fpath):
            subprocess.call(['datalad', 'get', filt4d_fpath])

        # load the time-series
        filt4d_img = nib.load(filt4d_fpath)
        filt4d_data = filt4d_img.get_fdata()
        # slice the a TR
        filt3d_data = filt4d_data[:, :, :, 20]

        # binarize the image
        ao_fov_bin = np.copy(filt3d_data)
        ao_fov_bin[ao_fov_bin < 1] = 0
        ao_fov_bin[ao_fov_bin >= 1] = 1

        # create a nifti image
        ao_fov_img = nib.Nifti1Image(ao_fov_bin,
                                     filt4d_img.affine,
                                     header=filt4d_img.header)

        # actual saving
        nib.save(ao_fov_img, out_fpath)

    else:
        ao_fov_img = nib.load(out_fpath)

    return ao_fov_img


def create_gray_matter_mask(subj, gray_matter_fpath, out_fpath):
    '''
    '''
    if not os.path.exists(out_fpath):
        gray_matter_img = nib.load(gray_matter_fpath)
        gray_matter_data = gray_matter_img.get_fdata()
        # following step probably unnecessary
        gray_matter_bin = np.copy(gray_matter_data)
        # binarize the data
        # DEPENDS ON THE CURRENT INPUT FILE (brain_seg.nii.gz)
        gray_matter_bin[gray_matter_bin != 2] = 0
        gray_matter_bin[gray_matter_bin == 2] = 1

        # prepare saving
        img = nib.Nifti1Image(gray_matter_bin,
                              gray_matter_img.affine,
                              header=gray_matter_img.header)
        # perform saving
        nib.save(img, out_fpath)

    else:
        img = nib.load(out_fpath)

    return img


def dilate_mask(in_file, out_file):
    '''
    '''
    if not os.path.exists(out_file):
        input_img = nib.load(in_file)
        # get data from image
        input_data = input_img.get_fdata()
        # dilate the data
        dil_data = scipy.ndimage.binary_dilation(input_data)
        # some cleaning
        dil_data[dil_data == False] = 0
        dil_data[dil_data == True] = 1

        # save the dilated mask
        new_img = nib.Nifti1Image(dil_data,
                                  input_img.affine,
                                  input_img.header)
        # save it
        nib.save(new_img, out_file)

    else:
        new_img = nib.load(out_file)

    return new_img


def count_voxels():
    '''
    '''


# main program #
if __name__ == "__main__":
    # find all masks that are available in MNI space
    masks_in_mni = find_files(os.path.join(ROIS_PATH, 'in_mni', '*.*'))

    # which mask are actually relevant (at the moment?)
    relevant_masks = [
        ('Group PPA', 'grp_PPA_bin.nii.gz'),
        ('Group FFA', 'grp_FFA_bin.nii.gz'),
        ('Group PPA (diluted)', 'grp_PPA_bin_dil.nii.gz'),
        ('Group FFA (diluted)', 'grp_FFA_bin_dil.nii.gz')
        # ('posterior parahippocampal g.', 'harv-oxf_prob_Parahippocampal Gyrus, posterior division.nii.gz'),
        # ('anterior temporal fusiform c.', 'harv-oxf_prob_Temporal Fusiform Cortex, anterior division.nii.gz'),
        # ('temp. occ. fusiform c.', 'harv-oxf_prob_Temporal Occipital Fusiform Cortex.nii.gz'),
        # ('lingual gyrus', 'harv-oxf_prob_Lingual Gyrus.nii.gz')
    ]

    # get all subjects from existing directories
    subjs_pathes = find_files(SUBJS_PATH_PATTERN)
    subjs = [re.search(r'sub-..', string).group() for string in subjs_pathes]
    # some filtering
    subjs = sorted(list(set(subjs)))

    for subj in subjs[:1]:
        # create the subject-specific folder in case it does not exist
        os.makedirs(os.path.join(subj, ROIS_PATH), exist_ok=True)

        # create (binarized) audio-description FoV mask
        # by slicing and binarizing a 4D image ('filtered_func_data.nii.gz')
        ao_pathes = [x for x in subjs_pathes if subj in x and 'audio' in x]
        # to get the audio FoV, we gonna look at run-1 (index = 0)
        input_4d_fpath = os.path.join(ao_pathes[0],
                                      'filtered_func_data.nii.gz')
        # the output file
        ao_out_fpath = os.path.join(subj,
                                    ROIS_PATH,
                                    'in_bold3Tp2',
                                    'audio_fov.nii.gz')

        # call the function that creates the mask, writes it to file,
        # and returns the mask as an image
        ao_fov_img = create_audio_FoV_mask(subj,
                                           input_4d_fpath,
                                           ao_out_fpath)

        # create the (binarized) gray matter mask
        gm_in_fpath = os.path.join(subj,
                                   ROIS_PATH,
                                   'in_bold3Tp2',
                                   'brain_seg.nii.gz')

        # create the output path
        in_file = os.path.basename(gm_in_fpath)
        out_fpath = gm_in_fpath.replace(in_file,
                                        'gm_bin.nii.gz')

        # call the function that creates the mask, writes it to file,
        # and returns the mask as an image
        gm_img = create_gray_matter_mask(subj,
                                         gm_in_fpath,
                                         out_fpath)

        # create a dilated mask by calling a function that dilates a mask
        # read from file, writes the dilated mask
        # and returns the dilated mask as an image
        in_fpath = out_fpath
        out_fpath = in_fpath.replace('gm_bin.nii.gz', 'gm_bin_dil.nii.gz')
        # get the image
        gm_dil_img = dilate_mask(in_fpath, out_fpath)


        # intersection of FoV & dilated mask
        gm_dil_in_fov = gm_dil_img.get_fdata() * ao_fov_img.get_fdata()

        # prepare saving
        img = nib.Nifti1Image(gm_dil_in_fov,
                              gm_img.affine,
                              gm_img.header)
        # save dilated gray matter in FoV
        out_fpath =  os.path.join(subj,
                                  ROIS_PATH,
                                  'in_bold3Tp2',
                                  'gm_bin_dil_fov.nii.gz')
        # save it
        nib.save(img, out_fpath)

        # following part counts voxels in some baseline areas
        # and voxels in an incrementally increased area of the brain by
        # merging single masks

        # AV 3D example from 4D 'filtered_func_data.nii.gz'
        # filter all subjects' runs (AO and AV) for current sub's AV
        av_pathes = [y for y in subjs_pathes if subj in y and 'movie' in y]
        # load one movie run as an example
        example_4d_fpath = os.path.join(av_pathes[0],
                                        'filtered_func_data.nii.gz')
        filt4d_img = nib.load(example_4d_fpath)
        filt3d_example = filt4d_img.get_fdata()[:, :, :, 20]
        # mask movie data with audio FoV
        movie_in_ao_fov = filt3d_example * ao_fov_img.get_fdata()

        print('Non-zero voxels in:')
        print('FoV of audio-description:\t',
              np.count_nonzero(ao_fov_img.get_fdata()))
        print('FoV overlayed on movie:\t\t',
              np.count_nonzero(movie_in_ao_fov))
        print('FoV overlayed on (dil.) gray m.:',
              np.count_nonzero(gm_dil_in_fov))

#         merged_masks = np.array([])
#         for name_n_fname in relevant_masks:
#
#             name, fname = name_n_fname
#
#             # get the complete path for the file
#             fpath = os.path.join(ROIS_PATH, subj, fname)
#
#             # load the file
#             mask_img = nib.load(fpath)
#             # get the actual data of the image
#             mask_data = mask_img.get_fdata()
#             # binarize the probabilistic map
#             mask_data[mask_data > 0] = 1
#
#             # prepare saving the binarized mask
#             img = nib.Nifti1Image(mask_data,
#                                     mask_img.affine,
#                                     mask_img.header)
#             # save it
#             # nib.save(img, fname.replace('.nii.gz', '_bin.nii.gz'))
#
#             # restrict current mask to FOV
#             mask_data_fov = mask_data * audio_fov_bin
#             ###########################################################
#             # check number of individual PPA voxels in FOV
#
#             # restrict current FOV to gray-matter
#             mask_data_fov_gm = mask_data_fov * gray_matter_bin_dil
#             ###########################################################
#             # check if number of individual PPA voxel in gray-matter mask
#
#             # count overall number of voxels of the current mask
#             # that remain after masking with FoV & gray matter
#             print(f'{name} has {np.count_nonzero(mask_data_fov_gm)}' +
#                     f' voxels (of {np.count_nonzero(mask_data)}) in' +
#                     ' GM of audio FoV')
#
#             if not merged_masks.any():
#                 merged_masks = mask_data_fov_gm
#                 continue
#
#             else:
#                 # create the new merged masks
#                 new_merged_masks = merged_masks + mask_data_fov_gm
#                 new_merged_masks[new_merged_masks > 0] = 1
#
#                 # current number of all voxels
#                 current_no_vox = np.count_nonzero(new_merged_masks)
#
#                 # count the number exclusively newly added voxels
#                 exclusively_new = mask_data_fov_gm - merged_masks
#                 exclusively_new[exclusively_new <= 0] = 0
#                 new_vox = np.count_nonzero(exclusively_new)
#
#                 print(f'{name} adds {new_vox} new voxels to previous' +
#                         f' {np.count_nonzero(merged_masks)}' +
#                         f' voxels (={current_no_vox})')
#
#                 # prepare for next loop
#                 merged_masks = new_merged_masks
#
#                 # add next masks to previous merged_masks_bin
#                 # binarize again
#
#                 # VOXEL-ANZAHL der individuellen PPA in merged Mask
#
#                 ##########################################################
#                 # VOXEL-ANZAHL der individuellen PPA,
#                 # die durch die exklusiv neuen Voxel dazukommen
#
#         print('\nFinal mask size:', np.count_nonzero(merged_masks), '\n\n')
#         # checken, ob alle Voxel in VIS (und AO) PPA innerhalb der Maske sind
#         # group PPA in subject space:
#         # e.g. '/studyforrest-paper-ppa/rois-and-masks/sub-01/grp_PPA_bin.nii.gz'
#         # group PPA in MNI:
#         # 'studyforrest-paper-ppa/rois-and-masks/bilat_PPA_prob.nii.gz'
#         # wie wurde die erstellt?
#


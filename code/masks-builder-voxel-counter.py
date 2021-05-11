#!/usr/bin/env python3
'''
author: Christian Olaf Häusler
created on Wednesday, 31 March 2021

ToDo:
    - change ouput path to 'sub-*/masks'
    - in line with BIDS structure?

'''

from glob import glob
import subprocess
import nibabel as nib
import numpy as np
import os
import re
import scipy.ndimage

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


def create_audio_FoV_mask(subj, out_fpath):
    '''
    '''
    # create the 3D image (non-binarized)
    if not os.path.exists(out_fpath):
        print('creating audio FoV mask for subject', subj)

        # filter pathes for current subject and stimulus
        # (subjs_pathes is global variable)
        ao_pathes = [x for x in subjs_pathes if subj in x and 'audio' in x]

        # to get the audio FoV, we gonna look in a 4D image and slice one TR
        filt4d_fpath = os.path.join(ao_pathes[0], 'filtered_func_data.nii.gz')

        # download data via datalad if needed
        if not os.path.exists(filt4d_fpath):
            subprocess.call(['datalad', 'get', filt4d_fpath])

        # DEBUGGING CHECKING
        print('input:', filt4d_fpath)
        print('output:', out_fpath)

        # load the time-series
        filt4d_img = nib.load(filt4d_fpath)
        filt4d_data = filt4d_img.get_fdata()

        # slice the a TR
        filt3d_data = filt4d_data[:, :, :, 20]

        # create a nifti image
        filt3d_img = nib.Nifti1Image(
            filt3d_data,
            filt4d_img.affine,
            header=filt4d_img.header
        )

        # save the image
        out_fpath = os.path.join(
            ROIS_PATH,
            subj,
            'audio_fov.nii.gz'
        )
        # actual saving
        nib.save(filt3d_img, out_fpath)

    else:
        filt3d_img = nib.load(out_fpath)

    return filt3d_img


def create_gray_matter_mask(subj, gray_matter_fpath):
    '''
    '''
    in_file = os.path.basename(gray_matter_fpath)
    out_fpath = gray_matter_fpath.replace(
        in_file,
        'gray_matter_bin.nii.gz')

    out_fpath = out_fpath.replace('/in_bold3Tp2', '')

    if not os.path.exists(out_fpath):
        gray_matter_img = nib.load(gray_matter_fpath)
        gray_matter_data = gray_matter_img.get_fdata()
        # following step probably unnecessary
        gray_matter_bin = np.copy(gray_matter_data)
        # binarize the data
        # DEPENDS ON THE CURRENT INPUT FILE
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


# main program #
if __name__ == "__main__":
    # some hardcoded sources
    # path of the subdataset providing templates and transformatiom matrices
    TNT_DIR = 'inputs/studyforrest-data-templatetransforms'

    # the path that contains mask (input and output purpose)
    ROIS_PATH = 'rois-and-masks'

    # the path to check for which subjects we have (filtered) functional data
    # that were used to localize the PPA via movie and audio-description
    SUBJS_PATH_PATTERN = 'inputs/studyforrest-ppa-analysis/sub-??/' + \
        'run-?_*-ppa-ind.feat'

    subjs_pathes = find_files(SUBJS_PATH_PATTERN)
    subjs = [re.search(r'sub-..', string).group() for string in subjs_pathes]
    # some filtering (which is probably not necessary)
    subjs = sorted(list(set(subjs)))

    masks_in_mni = find_files(os.path.join(ROIS_PATH, 'in_mni', '*.*'))

    for subj in subjs:
        # create the subject-specific folder in case it does not exist
        os.makedirs(os.path.join(ROIS_PATH, subj), exist_ok=True)

        # get AV 3D example from 4D 'filtered_func_data.nii.gz'
        audio_fov_fpath = os.path.join(
            ROIS_PATH,
            subj,
            'in_bold3Tp2',
            'audio_fov.nii.gz')

        # functions returns an 3D image of AO run-1
        # in case the file does not exists it creates the image from 4D data
        audio_fov_img = create_audio_FoV_mask(subj, audio_fov_fpath)
        # binarize the image
        audio_fov_data = audio_fov_img.get_fdata()
        audio_fov_bin = np.copy(audio_fov_data)
        audio_fov_bin[audio_fov_bin < 1] = 0
        audio_fov_bin[audio_fov_bin >= 1] = 1

        # create the binarized (!) gray matter mask
        gray_matter_fpath = os.path.join(
            ROIS_PATH,
            subj,
            'in_bold3Tp2',
            'brain_seg.nii.gz')  # preliminary input

        # functions returns the binarized image
        # and, if needed, creates the image on-the-fly
        gray_matter_bin_img = create_gray_matter_mask(subj, gray_matter_fpath)
        # get data from image
        gray_matter_bin_data = gray_matter_bin_img.get_fdata()
        # dilate the gray matter
        gray_matter_bin_dil = scipy.ndimage.binary_dilation(gray_matter_bin_data)
        gray_matter_bin_dil[gray_matter_bin_dil == False] = 0
        gray_matter_bin_dil[gray_matter_bin_dil == True] = 1

        # DEBUGGIN / CHECKING: save the dilated image
        img = nib.Nifti1Image(gray_matter_bin_dil,
                              gray_matter_bin_img.affine,
                              gray_matter_bin_img.header)
        # DEBUGGIN / CHECKING
        nib.save(img,
                 os.path.join(ROIS_PATH, subj, 'gray_matter_bin_dil.nii.gz'))

        # intersection of FoV & dilated mask
        gray_mat_dil_fov = gray_matter_bin_dil * audio_fov_bin

        # prepare saving
        img = nib.Nifti1Image(gray_mat_dil_fov,
                              gray_matter_bin_img.affine,
                              gray_matter_bin_img.header)
        # save
        nib.save(img,
                 os.path.join(ROIS_PATH,
                              subj,
                              'gray_matter_bin_dil_fov.nii.gz'
                              )
                 )

        # AV 3D example from 4D 'filtered_func_data.nii.gz'
        # filter all subjects' runs (AO and AV) for current sub's AV
        av_pathes = [y for y in subjs_pathes if subj in y and 'movie' in y]
        # load one movie run as an example
        example_4d_fpath = os.path.join(av_pathes[0],
                                        'filtered_func_data.nii.gz')
        filt4d_img = nib.load(example_4d_fpath)
        filt4d_data = filt4d_img.get_fdata()
        filt3d_movie_example = filt4d_data[:, :, :, 20]

        # DEBUGGIING / CHECKING: save the image

        # mask movie data with audio FoV
        movie_fov = filt3d_movie_example * audio_fov_bin

        print('Non-zero voxels in:')
        print('20th TR of movie run-1:\t',
              np.count_nonzero(filt3d_movie_example))
        print('FoV mask:\t\t',
              np.count_nonzero(audio_fov_bin))
        print('FoV applied to movie:\t',
              np.count_nonzero(movie_fov))
        print('(dil.) gray m. in FoV:\t',
              np.count_nonzero(gray_mat_dil_fov))
        print('(dil.) gray m. in FoV on movie data:',
              np.count_nonzero(gray_mat_dil_fov * filt3d_movie_example), '\n')

        relevant_masks = [
            ('Group PPA', 'grp_PPA_bin.nii.gz'),
            ('Group FFA', 'grp_FFA_bin.nii.gz'),
            ('Group PPA (diluted)', 'grp_PPA_bin_dil.nii.gz'),
            ('Group FFA (diluted)', 'grp_FFA_bin_dil.nii.gz')
#             ('posterior parahippocampal g.', 'harv-oxf_prob_Parahippocampal Gyrus, posterior division.nii.gz'),
#             ('anterior temporal fusiform c.', 'harv-oxf_prob_Temporal Fusiform Cortex, anterior division.nii.gz'),
#             ('temp. occ. fusiform c.', 'harv-oxf_prob_Temporal Occipital Fusiform Cortex.nii.gz'),
#             ('lingual gyrus', 'harv-oxf_prob_Lingual Gyrus.nii.gz')
        ]

        ####################################################################
        # VOXEL-ANZAHL in der individuellen PPA?

        merged_masks = np.array([])
        for name_n_fname in relevant_masks:

            name, fname = name_n_fname

            # get the complete path for the file
            fpath = os.path.join(ROIS_PATH, subj, fname)

            # load the file
            mask_img = nib.load(fpath)
            # get the actual data of the image
            mask_data = mask_img.get_fdata()
            # binarize the probabilistic map
            mask_data[mask_data > 0] = 1

            # prepare saving the binarized mask
            img = nib.Nifti1Image(mask_data,
                                  mask_img.affine,
                                  mask_img.header)
            # save it
            # nib.save(img, fname.replace('.nii.gz', '_bin.nii.gz'))

            # restrict current mask to FOV
            mask_data_fov = mask_data * audio_fov_bin
            ###########################################################
            # check number of individual PPA voxels in FOV

            # restrict current FOV to gray-matter
            mask_data_fov_gm = mask_data_fov * gray_matter_bin_dil
            ###########################################################
            # check if number of individual PPA voxel in gray-matter mask

            # count overall number of voxels of the current mask
            # that remain after masking with FoV & gray matter
            print(f'{name} has {np.count_nonzero(mask_data_fov_gm)}' +
                    f' voxels (of {np.count_nonzero(mask_data)}) in' +
                    ' GM of audio FoV')

            if not merged_masks.any():
                merged_masks = mask_data_fov_gm
                continue

            else:
                # create the new merged masks
                new_merged_masks = merged_masks + mask_data_fov_gm
                new_merged_masks[new_merged_masks > 0] = 1

                # current number of all voxels
                current_no_vox = np.count_nonzero(new_merged_masks)

                # count the number exclusively newly added voxels
                exclusively_new = mask_data_fov_gm - merged_masks
                exclusively_new[exclusively_new <= 0] = 0
                new_vox = np.count_nonzero(exclusively_new)

                print(f'{name} adds {new_vox} new voxels to previous' +
                      f' {np.count_nonzero(merged_masks)}' +
                      f' voxels (={current_no_vox})')

                # prepare for next loop
                merged_masks = new_merged_masks

                # add next masks to previous merged_masks_bin
                # binarize again

                # VOXEL-ANZAHL der individuellen PPA in merged Mask

                ##########################################################
                # VOXEL-ANZAHL der individuellen PPA,
                # die durch die exklusiv neuen Voxel dazukommen

        print('\nFinal mask size:', np.count_nonzero(merged_masks), '\n\n')
        # checken, ob alle Voxel in VIS (und AO) PPA innerhalb der Maske sind
        # group PPA in subject space:
        # e.g. '/studyforrest-paper-ppa/rois-and-masks/sub-01/grp_PPA_bin.nii.gz'
        # group PPA in MNI:
        # 'studyforrest-paper-ppa/rois-and-masks/bilat_PPA_prob.nii.gz'
        # wie wurde die erstellt?


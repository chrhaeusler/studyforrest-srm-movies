#!/usr/bin/env python3
'''
author: Christian Olaf Häusler
created on Wednesday, 31 March 2021

ToDo:
    - change ouput path to 'sub-*/masks'
    - in line with BIDS structure?

'''

from glob import glob
import argparse
import subprocess
import nibabel as nib
import numpy as np
import os
import re
import scipy.ndimage


def parse_arguments():
    '''
    '''
    parser = argparse.ArgumentParser(
        description='warps brain maps from MNI to bold3Tp (subject) space'
    )

    parser.add_argument('-i',
                        default='inputs/studyforrest-data-templatetransforms',
                        help='input directory containing templates' +
                        '& transformations')

    parser.add_argument('-o',
                        default='masks',
                        help='output directory')

    args = parser.parse_args()

    inDir = args.i
    outDir = args.o

    return inDir, outDir


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


def create_grp_bilat_mask(grp_masks, grp_out):
    '''
    '''
    for input_file in grp_masks:
        if not os.path.exists(input_file):
            subprocess.call(['datalad', 'get', input_file])

    mask_img = nib.load(grp_masks[0])
    mask_data = np.array(mask_img.dataobj)

    # if there is more than one mask, load all of them and take their sum
    if len(grp_masks) > 1:
        for mask in grp_masks[1:]:
            mask_data += np.array(nib.load(mask).dataobj)

    # save the probabilistic map
    bilat_mask_img = nib.Nifti1Image(mask_data,
                                     mask_img.affine,
                                     header=mask_img.header)
    nib.save(bilat_mask_img, grp_out)

    # make the data binary by setting all non-zeros to 1
    mask_data[mask_data > 0] = 1

    # save the binarized mask
    bilat_mask_img = nib.Nifti1Image(mask_data,
                                     mask_img.affine,
                                     header=mask_img.header)
    nib.save(bilat_mask_img, grp_out.replace('prob', 'binary'))

    return None


def grp_ppa_to_ind_space(input, output, ref, warp):
    '''
    '''
    if not os.path.exists(ref):
        subprocess.call(['datalad', 'get', ref])

    if not os.path.exists(warp):
        subprocess.call(['datalad', 'get', warp])

    subprocess.call(
        ['applywarp',
         '-i', input,
         '-o', output,
         '-r', ref,
         '-w', warp,
         # '--premat=premat'
         ])

    return None


def mni_2_ind_bold3Tp2(mni_mask, out_fpath, ind_ref, ind_warp, premat):
    '''
    '''
    if not os.path.exists(mni_mask):
        subprocess.call(['datalad', 'get', mni_mask])

    if not os.path.exists(ind_ref):
        subprocess.call(['datalad', 'get', ind_ref])

    if not os.path.exists(ind_warp):
        subprocess.call(['datalad', 'get', ind_warp])

    if not os.path.exists(premat):
        subprocess.call(['datalad', 'get', premat])

    if not os.path.exists(out_fpath):
        subprocess.call(
            ['applywarp',
             '-i', mni_mask,
             '-o', out_fpath,
             '-r', ind_ref,
             '-w', ind_warp,
             '--premat=' + premat
             ]
        )

    return None


# main program #
if __name__ == "__main__":
    # path of the subdataset providing templates and transformatiom matrices
    # the path that contains mask (input and output purpose)
    TNT_DIR, ROIS_PATH = parse_arguments()

    # filename for pre-transform (affine matrix)
    XFM_MAT = os.path.join(
        TNT_DIR,
        'templates/grpbold3Tp2/xfm/',
        'mni2tmpl_12dof.mat'
    )



    # create bilateral ROI overlaps (at the moment just focus on PPA & FFA)
    rois = ['PPA', 'FFA']
    # loop over rois that are about to merged
    for roi in rois:
        # create the input path/filename
        uni_grp_mask_pattern = os.path.join(
            ROIS_PATH,
            'in_mni',
            f'?{roi}_overlap.nii.gz'
        )
        # find all files (most subjects and all except primary visual area
        # have bilateral clusters/masks)
        uni_grp_mask_fpathes = find_files(uni_grp_mask_pattern)
        # output
        bilat_outfpath = os.path.join(
            ROIS_PATH,
            'in_mni',
            f'{roi}_overlap_prob.nii.gz'
        )
        # do the conversion
        create_grp_bilat_mask(uni_grp_mask_fpathes, bilat_outfpath)


    # warp probabilistic maps and FoV of audio-description to bold3Tp2
    # the path to check for which subjects we have (filtered) functional data
    # that were used to localize the PPA via movie and audio-description
    SUBJS_PATH_PATTERN = 'inputs/studyforrest-ppa-analysis/sub-??'
    subjs_pathes = find_files(SUBJS_PATH_PATTERN)
    subjs = [re.search(r'sub-..', string).group() for string in subjs_pathes]
    # some filtering
    subjs = sorted(list(set(subjs)))

    # find the masks to be warped in to subjects' spaces
    masks_in_mni = find_files(os.path.join(ROIS_PATH, 'in_mni', '*.*'))
    # filter for relevant masks
    # masks_overlap = [mask for mask in masks_in_mni if 'overlap' in mask]
    masks_in_mni = [mask for mask in masks_in_mni if 'overlap' not in mask]

    for subj in subjs[:1]:
        # create the subject-specific folder in case it does not exist
        out_path = os.path.join(subj, ROIS_PATH, 'in_bold3Tp2')
        os.makedirs(out_path, exist_ok=True)

        for mask_in_mni in masks_in_mni:
            # output path/filename
            out_file = os.path.basename(mask_in_mni)
            out_fpath = os.path.join(subj, ROIS_PATH, 'in_bold3Tp2', out_file)

            # the path of the (individual) reference image
            subj_ref = os.path.join(TNT_DIR, subj, 'bold3Tp2/brain.nii.gz')

            # the volume providing warp/coefficient
            subj_warp = os.path.join(
                TNT_DIR,
                subj,
                'bold3Tp2/in_grpbold3Tp2/'
                'tmpl2subj_warp.nii.gz'
            )

            # the (affine) pre-transformation matrix
            premat = XFM_MAT

            # do the warping from MNI to bold3Tp2 by calling the function
            mni_2_ind_bold3Tp2(
                mask_in_mni,
                out_fpath,
                subj_ref,
                subj_warp,
                premat
            )

            # warp the (bilateral) union of PPA ROIS that were created above
            # from MNI to subject space
            for roi in rois:
                # create the names of pathes
                in_fpath = os.path.join(
                    ROIS_PATH,
                    'in_mni',
                    f'{roi}_overlap_binary.nii.gz'
                )

                out_fpath = os.path.join(
                    subj,
                    ROIS_PATH,
                    'in_bold3Tp2',
                    f'grp_{roi}_bin.nii.gz'
                )

                # call the function to warp from MNI to bold3Tp2
                grp_ppa_to_ind_space(in_fpath, out_fpath, subj_ref, subj_warp)

                # binarize the mask and overwrite existing file
                in_fpath = out_fpath
                roi_img = nib.load(in_fpath)
                roi_data = roi_img.get_fdata()
                # perform the actual binarization
                roi_data[roi_data > 0] = 1
                # create image from data array
                img = nib.Nifti1Image(roi_data,
                                      roi_img.affine,
                                      roi_img.header)
                # save it
                nib.save(img, out_fpath)

                # dilate the union of ROIs by one voxel
                in_fpath = out_fpath
                out_fpath = in_fpath.replace('.nii.gz', '_dil.nii.gz')
                roi_img = nib.load(in_fpath)
                roi_data = roi_img.get_fdata()

                # perform the acutal dilations
                roi_data_dil = scipy.ndimage.binary_dilation(roi_data)
                roi_data_dil[roi_data_dil == False] = 0
                roi_data_dil[roi_data_dil == True] = 1

                # prepare saving
                img = nib.Nifti1Image(roi_data_dil,
                                      roi_img.affine,
                                      roi_img.header)
                # save
                nib.save(img, out_fpath)


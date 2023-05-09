#!/usr/bin/env python3
'''
created on Mon May 17th 2021
author: Christian Olaf Haeusler
'''

from glob import glob
# from nilearn.input_data import NiftiMasker, MultiNiftiMasker
import matplotlib.pyplot as plt
import brainiak.funcalign.srm
import argparse
# import nibabel as nib
import numpy as np
import os
import re
import scipy.spatial.distance as sp_distance


# do some slicing
start = 0
end = 7747  # first quarter of audio-description


def parse_arguments():
    '''
    '''
    parser = argparse.ArgumentParser(
        description='load a pickled SRM and do some plotting'
    )

    parser.add_argument('-sub',
                        required=False,
                        default='sub-01',
                        help='subject to leave out (e.g. "subj-01")')

    parser.add_argument('-indir',
                        required=False,
                        default='test',
                        help='output directory (e.g. "sub-01")')

    parser.add_argument('-nfeat',
                        required=False,
                        default='10',
                        help='number of features (shared responses)')

    parser.add_argument('-niter',
                        required=False,
                        default='30',
                        help='number of iterations')

    parser.add_argument('-outdir',
                        required=False,
                        default='test',
                        help='name ouf output directory')


    args = parser.parse_args()

    sub = args.sub
    indir = args.indir
    n_feat = int(args.nfeat)
    n_iter = int(args.niter)
    outdir = args.outdir

    return sub, indir, n_feat, n_iter, outdir


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


def plot_feat_x_timepoints(srm):
    '''Plot the shared response
    '''
    print('SRM: Features X Time-points ', srm.s_.shape)
    plt.figure(figsize=(15, 4))
    plt.title('SRM: Features X Time-points')
    plt.xlabel('TR')
    plt.ylabel('feature')
    plt.yticks(list(range(0, n_feat)))
    plt.hlines([y + 0.5 for y in (range(0, n_feat))], 0, srm.s_.shape[1],
               colors='k', linewidth=.5, linestyle='dashed')
    plt.imshow(srm.s_, cmap='viridis', aspect='auto')
    plt.tight_layout()
    plt.colorbar()
    # save it
    plt.savefig(f'test/features{n_feat}_time-points.svg', bbox_inches='tight')
    plt.close()

    return None


def plot_top_repsonses(srm, start, end):
    '''
    '''
    plt.figure(figsize=(15, 4))

    # title
    # plt.title('Time series of shared features')
    # do some slicing, so the plot does not get too crowded
    plt.plot(srm.s_[0, start:end], linewidth=0.5)
    plt.plot(srm.s_[1, start:end], linewidth=0.5)
    plt.plot(srm.s_[2, start:end], linewidth=0.5)
#     plt.plot(srm.s_[3, start:end], linewidth=0.5)
#     plt.plot(srm.s_[4, start:end], linewidth=0.5)
#     plt.plot(srm.s_[5, start:end], linewidth=0.5)
#     plt.plot(srm.s_[6, start:end], linewidth=0.5)
#     plt.plot(srm.s_[7, start:end], linewidth=0.5)
#     plt.plot(srm.s_[8, start:end], linewidth=0.5)
#     plt.plot(srm.s_[9, start:end], linewidth=0.5)

    # some "making it neat"
    plt.xlim(start, end)

    # label x axis
    plt.xlabel('TR')

    extensions = ['pdf', 'png', 'svg']
    for extension in extensions:
        fpath = os.path.join(outDir, f'srm-time-series.{extension}')
        plt.savefig(fpath, bbox_inches='tight')

    return None


def plot_distance_mtrx(srm, start, end):
    '''
    '''
    matrixFpath = os.path.join(outDir, 'distance-matrix.npy')

    if not os.path.isfile(matrixFpath):
        print('file does not exist')
        dist_mat = sp_distance.squareform(sp_distance.pdist(srm.s_[:, start:end].T))
        np.save(matrixFpath, dist_mat)
    else:
        print('file exists')
        dist_mat = np.load(matrixFpath)

    plt.figure(figsize=(21, 15))
    plt.title('Distance between pairs of time points in shared space')
    plt.xlabel('TR')
    plt.ylabel('TR')
    plt.imshow(dist_mat, cmap='viridis')
    plt.colorbar()

    extensions = ['pdf', 'png', 'svg']
    for extension in extensions:
        fpath = os.path.join(outDir, f'distance-matrix.{extension}')
        plt.savefig(fpath, bbox_inches='tight')

    return dist_mat


if __name__ == "__main__":
    # read command line arguments
    subj, in_dir, n_feat, n_iter, outDir = parse_arguments()

    # create output directory
    os.makedirs(outDir, exist_ok=True)

    # save model as (zipped) pickle variable
    in_fpath = os.path.join(
        in_dir, subj, f'srm-ao-av-vis_feat{n_feat}-iter{n_iter}.npz'
    )

    # load the srm from file
    srm = load_srm(in_fpath)

    # plot depicting features x timepoints
    plot_feat_x_timepoints(srm)

    # plot depicting time-series of x top shared responses
    plot_top_repsonses(srm, start, end)

    # plot distance matrix
    dist_mat = plot_distance_mtrx(srm, start, end)

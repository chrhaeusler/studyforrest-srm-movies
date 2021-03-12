#!/usr/bin/env python3
'''
created on Fri March 12th 2021
author: Christian Olaf Haeusler

To Do:
    - google.docs: Plan lesen und Fragen an Michael notieren?

- wie auf Lisa und Susanne zugehen?
- wie viele Daten sind in welchem Format benutzbar?

'''

import argparse
# from glob import glob
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import nibabel as nib
# from nilearn import plotting
# from nilearn.image import smooth_img
# import os
# import re
# import subprocess


def parse_arguments():
    '''
    '''
    parser = argparse.ArgumentParser(
        description='Create figures of results'
    )

    parser.add_argument('-o',
                        required=False,
                        default='paper/figures',
                        help='the folder where the figures are written into')

    args = parser.parse_args()

    outPath = args.o

    return outPath


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


if __name__ == "__main__":
    # read command line arguments
    outPath = parse_arguments()


    # download Movies Study data
    # we need all movies data (not just Forrest Snippet) to create the common
    # model space (-> vgl. Olivers github)
    # probably, we will not use the resting state data

    # preprocessing
    # https://fmriprep.org
    # cf. Jean's scripts (that should be tailored to RS data)
    # what needs (not) to be done during processing?



    # Common Model Space + Movies VPN
    # https://brainiak.org/
    # goal:
    # - create CMS
    # - get transformation matrices to align movies VPN to "their" CMS
    #   (any later usage for transformation matrices of movies VPN?)
    # input:
    # - all movie runs of all subjects (maybe exclude 'bad listeners')
    # process:
    # - magically create CMS
    # output:
    # 1) common model space (CMS):     time points x components
    # 2) subspace (9.5m Forrest):      time points x components
    # 3) transformations (movies VP):  components (weights) x voxel


    # download Forrest data
    # which ones? as usual, "aligned" from github?
    # get fmri data for audio-only and audiovisual movie
    # datalad install -d . -s https://github.com/psychoinformatics-de/studyforrest-data-aligned inputs/studyforrest-data-aligned

    # get 4D fMRI data and motion correction parameters
    # at the beginning, we just need run-1 cause it contains the ~9.5 min that
    # are also in the movies study
    # datalad get inputs/studyforrest-data-aligned/sub-??/in_bold3Tp2/sub-??_task-a?movie_run-?_bold*.*

    # any further pre-processing of the Forrest Data?



    ### here comes the pain ##################################################
    # Common Model Space + Forrest VPN
    # goal:
    # - get transformation matrices of Forrest VPN in order to later XYZ?
    # input: 9 minutes from first run (time points x voxel)
    # process: do what here?
    # output:
    # -
    # -

    ##########################################################################
    # align "noisy" studyforrest VPN data (time points x voxel; Forrest snippet)
    # to denoised Movies CMS (time points x components)
    # HOW?

    ###########################################################################
    # ...and I do not get following part anymore:
    # “sub common model space” = individual data x transformation matrix
    # time points x components = (time points x voxel) x (voxel x components)
    # min (dist (components - data))
    # aber: Abstände sollen minimiert werden, indem auf Template angeglichen wird und nicht, indem sich beide “in der Mitte treffen” (dunno if this makes sense)



    # Testing
    # visual localizer von FG benutzen, um visual areas in anderen VPN
    # vorherzusagen
    # Vorteil: Commond Model Space bleibt immer der gleiche; VPN sind
    # unabhängig
    # Nachteil: Nicht besonders neu, aber bisschen neu reicht; "partial" times
    # series alignment
    # Stimulus-Bindeglied mit Time-Series SRM ist neu
    # aber

    # Scatch:
    # Hörspiel-Daten & Auditorische PPA?
    # BOLD-Kontraste vs. "Component-Kontraste"?
    # voxel-wise encoding -> Sprach-Annot für Movies Study (im common space
    # gehen auch voxelwise encoding, sind dann aber component wise)


    print('End of Script')

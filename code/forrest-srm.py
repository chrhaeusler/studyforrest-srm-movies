#!/usr/bin/env python3
"""
created on Mon March 29th 2021
author: Christian Olaf Haeusler

To Do:
    - well, everything

"""

import argparse
from glob import glob
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import nibabel as nib
# from nilearn import plotting
# from nilearn.image import smooth_img
# import os
# import re
# import subprocess


def parse_arguments():
    """
    """
    parser = argparse.ArgumentParser(
        description="Create figures of results"
    )

    parser.add_argument("-o",
                        required=False,
                        default="paper/figures",
                        help="the folder where the figures are written into")

    args = parser.parse_args()

    outPath = args.o

    return outPath


def find_files(pattern):
    """
    """
    found_files = glob(pattern)
    found_files = sort_nicely(found_files)

    return found_files


def sort_nicely(l):
    """Sorts a given list in the way that humans expect
    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    l.sort(key=alphanum_key)

    return l


if __name__ == "__main__":
    # read command line arguments
    outPath = parse_arguments()




    print("End of Script")

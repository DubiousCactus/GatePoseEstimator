#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 theo <theo@not-arch-linux>
#
# Distributed under terms of the MIT license.

import numpy as np

from PIL import Image


"""
Several image manipulation utility functions
"""

'''
    Crops the given image over the given bounding box coordinates, and applies
    zero-padding to the rest of the image. The returned image in fact has the
    same dimensions as the given image
'''
def crop_and_pad(img, corner_min, corner_max):
    cropped = np.zeros(img.shape, dtype=img.dtype)
    cropped[
        corner_min[1]:corner_max[1],
        corner_min[0]:corner_max[0],:
    ] = img[corner_min[1]:corner_max[1], corner_min[0]:corner_max[0],:]
    assert cropped.shape == img.shape, "Cropped image has been resized!"
    return cropped

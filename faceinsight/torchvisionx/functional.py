# -*- coding: utf8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

from __future__ import division
import os
import random
from PIL import Image, ImageFilter


def gaussian_blur(img, radius=1.0):
    """Gaussian filter.
    
    Parameters
    ----------
    img : PIL image
        Input image (grayscale or color) to filter.
    radius : scalar, optional

    Returns
    -------
    filtered_image : PIL image

    """
    if radius < 0.0:
        raise(ValueError('Radius value less than zero is not valid.'))
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


# -*- coding: utf8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import random
import .functional as F

class RandomGaussianBlur(object):
    """Randomly Gaussian-blur the image with a random radius (default: of 1.0 to
    2.0).

    """
    def __init__(self, radius_range=(1.0, 2.0)):
        if radius_range[0]>radius_range[1]:
            raise(RuntimeError('range should be kind (min, max)'))
        
        self.radius_range = radius_range

    def __call__(self, img):
        """
        Args:
            img (PIL image): Image to be blurred.

        Returns:
            PIL Image: Randomly blurred image.
        """
        radius = random.uniform(*self.radius_range)
        return F.gaussian_blur(img, radius)

    def __repr__(self):
        return self.__class__.__name__ + '(radius in {0})'.format(self.radius_range)


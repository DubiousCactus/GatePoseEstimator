#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 theo <theo@not-arch-linux>
#
# Distributed under terms of the MIT license.

"""
Different utility functions used for training and testing.
"""

from keras.preprocessing.image import ImageDataGenerator

class GateGenerator(ImageDataGenerator):
    def flow_from_directory(self, config):
        pass

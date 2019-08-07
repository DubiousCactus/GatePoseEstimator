#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 theo <theo@not-arch-linux>
#
# Distributed under terms of the MIT license.

"""
Training script for the gate distance and rotation estimator.
"""

import yaml

from keras import backend as K

class Trainer:
    def __init__(self, config):
        with open(config, 'r') as config_file:
            try:
                self.config = yaml.safe_load(config_file)
            except yaml.YAMLError as exc:
                raise Exception(exc)


    def _get_model(self):
        # TODO: Return CNN or MLP and init Keras


    def train(self):
        # TODO: Load data generators, calc steps and set callbacks

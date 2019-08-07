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

import argparse
import models
import losses
import yaml

from utils import GateGenerator
from keras import backend as K
from keras.optimizers import Adam

class Trainer:
    def __init__(self, config):
        with open(config, 'r') as config_file:
            try:
                self.config = yaml.safe_load(config_file)
            except yaml.YAMLError as exc:
                raise Exception(exc)

        self.model = self._get_model()
        self.training_data = GateGenerator(self.config)
        self.training_data = self.training_data.flow_from_directory(self.config)
        self.validation_data = GateGenerator(self.config)
        self.validation_data = self.validation_data.flow_from_directory(self.config)

    def _get_model(self):
        model = self.config['model'] is 'CNN' ? models.CNN() : models.MLP()
        adam = Adam()
        model.compile(optimizer=adam, loss=losses.combined_loss, metrics=['accuracy'])

        return model

    def train(self):
        # TODO: Load data generators, calc steps and set callbacks



if __name__ == '__main__':
    K.clear_session()
    parser = argparse.ArgumentParser(description='''Training script for the gate
                                     distance and rotation detector''')
    parser.add_argument('--config', type=str, help='''Path to the YAML config
                        file''')
    args = parser.parse_args()
    trainer = Trainer(args)
    trainer.train()

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
import models
import losses
import argparse

from keras import backend as K
from utils import GateGenerator
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
        model = (models.CNN(self.config) if self.config['model'] is 'CNN'
                    else models.MLP(self.config))
        adam = Adam()
        combined_loss = losses.combined_loss(alpha=self.config['alpha'],
                                             beta=self.config['beta'])
        model.compile(optimizer=adam, loss=combined_loss, metrics=['accuracy'])

        return model

    def train(self):
        # TODO: Load data generators, calc steps and set callbacks
        pass



if __name__ == '__main__':
    K.clear_session()
    parser = argparse.ArgumentParser(description='''Training script for the gate
                                     distance and rotation detector''')
    parser.add_argument('--config', type=str, help='''Path to the YAML config
                        file''', required=True)
    args = parser.parse_args()
    trainer = Trainer(args.config)
    trainer.train()

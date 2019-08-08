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

    def _get_model(self):
        model = (models.CNN(self.config) if self.config['model'] is 'CNN'
                    else models.MLP(self.config))
        adam = Adam()
        combined_loss = losses.combined_loss(alpha=self.config['alpha'],
                                             beta=self.config['beta'])
        model.compile(optimizer=adam, loss=combined_loss, metrics=['accuracy'])

        return model

    def train(self):
        training_data_gen = GateGenerator()
        training_data = training_data_gen.flow_from_directory(
            self.config['training_dataset_root'],
            self.config['input_shape'],
            self.config['batch_size'],
            shuffle=True,
            ground_truth_available=True)

        validation_data_gen = GateGenerator()
        validation_data = validation_data_gen.flow_from_directory(
            self.config['validation_dataset_root'],
            self.config['input_shape'],
            self.config['batch_size'],
            shuffle=False,
            ground_truth_available=True)



if __name__ == '__main__':
    K.clear_session()
    parser = argparse.ArgumentParser(description='''Training script for the gate
                                     distance and rotation detector''')
    parser.add_argument('--config', type=str, help='''Path to the YAML config
                        file''', required=True)
    args = parser.parse_args()
    trainer = Trainer(args.config)
    trainer.train()

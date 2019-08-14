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
import argparse
import numpy as np

from keras import backend as K
from utils import GatePoseGenerator
from models import GatePoseEstimator
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard


class Trainer:
    def __init__(self, config):
        with open(config, 'r') as config_file:
            try:
                self.config = yaml.safe_load(config_file)
            except yaml.YAMLError as exc:
                raise Exception(exc)

        self.log_dir = args.log_dir
        self.model = self._get_model()

    def _get_model(self):
        model = GatePoseEstimator.build(self.config['training_target'],
                                        self.config['input_shape'])
        adam = Adam()
        model.compile(optimizer=adam, loss='mean_squared_error',
                      metrics=['accuracy'])

        return model

    def train(self):
        # TODO: Model restoring
        initial_epoch = 0

        training_data_gen = GatePoseGenerator(rescale=1./255)
        training_generator = training_data_gen.flow_from_directory(
            self.config['training_dataset_root'],
            self.config['image_shape'],
            self.config['input_shape'],
            self.config['training_target'],
            self.config['batch_size'],
            shuffle=True,
            ground_truth_available=True)

        validation_data_gen = GatePoseGenerator(rescale=1./255)
        validation_generator = validation_data_gen.flow_from_directory(
            self.config['validation_dataset_root'],
            self.config['image_shape'],
            self.config['input_shape'],
            self.config['training_target'],
            self.config['batch_size'],
            shuffle=False,
            ground_truth_available=True)

        steps_per_epoch = int(np.ceil(training_generator.samples /
                                      self.config['batch_size']))
        validation_steps = int(np.ceil(validation_generator.samples /
                                      self.config['batch_size']))

        early_stopping = EarlyStopping(monitor='val_loss',
                                       min_delta=0.0,
                                       patience=10,
                                       verbose=1)

        reduce_learning_rate = ReduceLROnPlateau(monitor='loss',
                                                 factor=0.2,
                                                 patience=6,
                                                 verbose=1,
                                                 epsilon=0.001,
                                                 cooldown=0,
                                                 min_lr=0.0000001)

        tensor_board = TensorBoard(log_dir=self.log_dir, histogram_freq=0,
                                   write_graph=True, write_images=True)

        self.model.fit_generator(training_generator,
                                 epochs=self.config['epochs'],
                                 steps_per_epoch=steps_per_epoch,
                                 callbacks=[early_stopping,
                                            reduce_learning_rate,
                                            tensor_board],
                                 validation_data=validation_generator,
                                 validation_steps=validation_steps,
                                 initial_epoch=initial_epoch)



if __name__ == '__main__':
    K.clear_session()
    parser = argparse.ArgumentParser(description='''Training script for the gate
                                     distance and rotation detector''')
    parser.add_argument('--config', type=str, help='''Path to the YAML config
                        file''', required=True)
    parser.add_argument('--log-dir', type=str, default='logs', help='''Path to
                        the logs directory file''')
    args = parser.parse_args()
    trainer = Trainer(args.config)
    trainer.train()

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

import os
import yaml
import models
import argparse
import numpy as np

from keras_radam import RAdam
from keras import backend as K
from keras.optimizers import Adam
from utils import GatePoseGenerator
from models import GatePoseEstimator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint


class Trainer:
    rot_acc_threshold = 0.03 # Difference threshold for the rotation accuracy
                             # computation, in degrees
    dist_acc_threshold = 0.25 # Difference threshold for the distance accuracy
                              # computation, in meters

    def __init__(self, config):
        with open(config, 'r') as config_file:
            try:
                self.config = yaml.safe_load(config_file)
            except yaml.YAMLError as exc:
                raise Exception(exc)

        self.log_dir = args.log_dir
        self.model = self._get_model(args.transfer_weights, args.fine_tune)


    @staticmethod
    def rotation_accuracy(y_true, y_pred):
        validate_el = lambda e: K.switch(np.abs(e[0]) < Trainer.rot_acc_threshold,
                                         lambda : 1.0, lambda : 0.0)
        valid_els = K.map_fn(validate_el, y_true - y_pred, name='accuracy')
        return K.mean(valid_els)

    @staticmethod
    def distance_accuracy(y_true, y_pred):
        validate_el = lambda e: K.switch(np.abs(e[0]) < Trainer.dist_acc_threshold,
                                         lambda : 1.0, lambda : 0.0)
        valid_els = K.map_fn(validate_el, y_true - y_pred, name='accuracy')
        return K.mean(valid_els)

    def _get_model(self, transfer_weights=None, fine_tune=False):
        model = GatePoseEstimator.build(self.config['training_target'],
                                        self.config['input_shape'], fine_tune)
        if transfer_weights:
            print("[*] Transfering weights from '{}'".format(transfer_weights))
            model.load_weights(transfer_weights, by_name=True, skip_mismatch=True)
        if self.config['training_target'] == 'distance':
            radam = RAdam()
            # radam = RAdam(total_steps=10000, warmup_proportion=0.1, min_lr=1e-5)
            model.compile(optimizer=radam, loss='mse',
                          metrics=['mae', Trainer.distance_accuracy])
        else:
            # radam = RAdam(total_steps=10000, warmup_proportion=0.1, min_lr=1e-5)
            adam = Adam(lr=0.01)
            model.compile(optimizer=adam, loss='mse',
                          metrics=['mae', Trainer.rotation_accuracy])

        return model

    def train(self):
        initial_epoch = self.config['initial_epoch']

        training_data_gen = GatePoseGenerator(rescale=1./255)#,
                                              # rotation_range=20,
                                              # channel_shift_range=0.5)
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
                                       patience=20,
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

        checkpoint = ModelCheckpoint(os.path.join(self.log_dir,
                                                  "epoch-{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5"),
                                     monitor='val_loss', verbose=1,
                                     save_best_only=True,
                                     save_weights_only=True)

        self.model.fit_generator(training_generator,
                                 epochs=self.config['epochs'],
                                 steps_per_epoch=steps_per_epoch,
                                 callbacks=[#early_stopping,
                                            checkpoint,
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
    parser.add_argument('--fine-tune', action='store_true', help='''Whether to
                        freeze the feature extraction layers for fine-tuning of
                        the fully connected layers''')
    parser.add_argument('--transfer-weights', type=str, default=None,
                        help='''Path to the weights file to transfer''')
    args = parser.parse_args()
    trainer = Trainer(args.config)
    trainer.train()

#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 theo <theo@not-arch-linux>
#
# Distributed under terms of the MIT license.

"""
Evaluation script for the gate distance and rotation estimator.
"""

import os
import yaml
import json
import models
import argparse
import numpy as np

from keras import backend as K
from utils import GatePoseGenerator
from models import GatePoseEstimator


class Evaluator:
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

        self.model = self._get_model(args.weights_path)


    @staticmethod
    def rotation_accuracy(y_true, y_pred):
        validate_el = lambda e: K.switch(np.abs(e[0]) < Evaluator.rot_acc_threshold,
                                         lambda : 1.0, lambda : 0.0)
        valid_els = K.map_fn(validate_el, y_true - y_pred, name='accuracy')
        return K.mean(valid_els)

    @staticmethod
    def distance_accuracy(y_true, y_pred):
        validate_el = lambda e: K.switch(np.abs(e[0]) < Evaluator.dist_acc_threshold,
                                         lambda : 1.0, lambda : 0.0)
        valid_els = K.map_fn(validate_el, y_true - y_pred, name='accuracy')
        return K.mean(valid_els)

    def _get_model(self, weights_path):
        model = GatePoseEstimator.build(self.config['training_target'],
                                        self.config['input_shape'])
        print("[*] Loading weights from '{}'".format(weights_path))
        model.load_weights(weights_path, by_name=True, skip_mismatch=True)
        if self.config['training_target'] == 'distance':
            model.compile(optimizer='adam', loss='mse',
                          metrics=['mae', Evaluator.distance_accuracy])
        else:
            model.compile(optimizer='adam', loss='mse',
                          metrics=['mae', Evaluator.rotation_accuracy])

        return model

    def eval(self):
        evaluation_data_gen = GatePoseGenerator(rescale=1./255)
        evaluation_generator = evaluation_data_gen.flow_from_directory(
            self.config['test_dataset_root'],
            self.config['image_shape'],
            self.config['input_shape'],
            self.config['training_target'],
            self.config['batch_size'],
            shuffle=False,
            ground_truth_available=True)

        steps_per_epoch = int(np.ceil(evaluation_generator.samples /
                                      self.config['batch_size']))
        metrics = self.model.evaluate_generator(evaluation_generator,
                                                steps=steps_per_epoch,
                                                workers=12,
                                                verbose=1)
        print("[*] MSE: {}".format(metrics[0]))
        print("[*] MAE: {}".format(metrics[1]))
        print("[*] Accuracy (<0.25m): {}".format(metrics[2]))


if __name__ == '__main__':
    K.clear_session()
    parser = argparse.ArgumentParser(description='''Training script for the gate
                                     distance and rotation detector''')
    parser.add_argument('--config', type=str, help='''Path to the YAML config
                        file''', required=True)
    parser.add_argument('--weights-path', type=str, help='''Path to the h5
                        weights file for the model''', required=True)
    args = parser.parse_args()
    evaluator = Evaluator(args.config)
    evaluator.eval()

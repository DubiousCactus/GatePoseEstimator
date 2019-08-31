#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 theo <theo@not-arch-linux>
#
# Distributed under terms of the MIT license.

"""
CNN and MLP branches
"""

import tensorflow as tf

from keras import regularizers
from keras.models import Model
from img_utils import crop_and_pad
from keras.layers import BatchNormalization, Dropout, Activation
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Lambda


class GatePoseEstimator:
    @staticmethod
    def build_rotation_branch(inputs):
        x = Conv2D(16, (3,3), padding="same", use_bias=False)(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(3, 3))(x)

        x = Conv2D(32, (3,3), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(3, 3))(x)

        x = Conv2D(64, (3,3), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Flatten()(x)
        x = Dropout(0.5)(x)

        x = Dense(16, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Dropout(0.5)(x)

        x = Dense(1, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu', name='rotation_output')(x)

        return x

    @staticmethod
    def build_rotation_dist_branch(inputs):
        x = Conv2D(16, (3,3), padding="same", use_bias=False)(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(3, 3))(x)

        x = Conv2D(32, (3,3), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(3, 3))(x)

        x = Conv2D(64, (3,3), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Flatten()(x)

        rot = Dense(16, use_bias=False)(x)
        rot = BatchNormalization()(rot)
        rot = Activation('relu')(rot)

        rot = Dense(1, use_bias=False)(rot)
        rot = BatchNormalization()(rot)
        rot = Activation('relu', name='rotation_output')(rot)

        dist = Dense(16, use_bias=False)(x)
        dist = BatchNormalization()(x)
        dist = Activation('relu')(x)

        dist = Dense(1, use_bias=False)(dist)
        dist = BatchNormalization()(dist)
        dist = Activation('relu', name='distance_output')(dist)

        return [rot, dist]
    '''
    Use only the bounding box coordinates as input for an MLP for the
    distance estimation.
    '''
    @staticmethod
    def build_distance_branch(inputs):
        x = Dense(100, use_bias=False)(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Dense(100, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Dense(1, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu', name='distance_output')(x)

        return x

    @staticmethod
    def build(model, shape, final_act='relu'):
        if model not in ['distance', 'rotation', 'combined']:
            raise ValueError('Model must be either "distance" or "rotation" or "combined"')
        if model == 'distance':
            branch_input = Input(shape=(4,), name='bbox_input')
            branch_output = GatePoseEstimator.build_distance_branch(branch_input)
        elif model == 'combined':
            branch_input = Input(shape=shape, name='img_input')
            branch_output = GatePoseEstimator.build_rotation_dist_branch(branch_input)
        else:
            branch_input = Input(shape=shape, name='img_input')
            branch_output = GatePoseEstimator.build_rotation_branch(branch_input)

        model = Model(inputs=branch_input, outputs=branch_output,
                      name='GatePoseEstimator')
        print(model.summary())

        return model

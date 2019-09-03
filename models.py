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
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Lambda, ReLU


class GatePoseEstimator:
    @staticmethod
    def build_rotation_branch(inputs, fine_tune):
        x = Conv2D(16, (3,3), padding="same", use_bias=False, trainable=(not fine_tune))(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(3, 3))(x)

        x = Conv2D(32, (3,3), padding='same', use_bias=False, trainable=(not fine_tune))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(3, 3))(x)

        x = Conv2D(64, (3,3), padding='same', use_bias=False, trainable=(not fine_tune))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(3, 3))(x)

        x = Conv2D(128, (3,3), padding='same', use_bias=False, trainable=(not fine_tune))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Flatten()(x)
        # x = Dropout(0.5)(x)

        # x = Dense(64, use_bias=False)(x)
        # x = BatchNormalization()(x)
        # x = Activation('linear')(x)

        # x = Dropout(0.5)(x)

        x = Dense(32, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # x = Dropout(0.5)(x)

        x = Dense(16, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        # x = Dropout(0.5)(x)

        x = Dense(1, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('tanh', name='rotation_output')(x)

        return x

    @staticmethod
    def buld_distance_branch(inputs, fine_tune):
        x = Conv2D(16, (3,3), padding="same", use_bias=False, trainable=(not fine_tune))(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(3, 3))(x)

        x = Conv2D(32, (3,3), padding='same', use_bias=False, trainable=(not fine_tune))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(3, 3))(x)

        x = Conv2D(64, (3,3), padding='same', use_bias=False, trainable=(not fine_tune))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(3, 3))(x)

        x = Conv2D(128, (3,3), padding='same', use_bias=False, trainable=(not fine_tune))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Flatten()(x)
        x = Dropout(0.5)(x)

        x = Dense(32, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # x = Dropout(0.5)(x)

        x = Dense(16, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        # x = Dropout(0.5)(x)

        x = Dense(1, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu', name='distance_output')(x)

        return x

    @staticmethod
    def build_rotation_dist_branch(inputs, fine_tune):
        x = Conv2D(16, (3,3), padding="same", use_bias=False, trainable=(not fine_tune))(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(3, 3))(x)

        x = Conv2D(32, (3,3), padding='same', use_bias=False, trainable=(not fine_tune))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(3, 3))(x)

        x = Conv2D(64, (3,3), padding='same', use_bias=False, trainable=(not fine_tune))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Flatten()(x)
        x = Dropout(0.5)(x)

        rot = Dense(16, use_bias=False)(x)
        rot = BatchNormalization()(rot)
        rot = Activation('relu')(rot)
        rot = Dropout(0.5)(rot)

        rot = Dense(1, use_bias=False)(rot)
        rot = BatchNormalization()(rot)
        rot = Activation('relu', name='rotation_output')(rot)

        dist = Dense(16, use_bias=False)(x)
        dist = BatchNormalization()(x)
        dist = Activation('relu')(x)
        dist = Dropout(0.5)(dist)

        dist = Dense(1, use_bias=False)(dist)
        dist = BatchNormalization()(dist)
        dist = Activation('relu', name='distance_output')(dist)

        return [rot, dist]

    @staticmethod
    def build(model, shape, fine_tune):
        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        # tf.keras.backend.set_session(tf.Session(config=config))
        if model not in ['distance', 'rotation', 'combined']:
            raise ValueError('Model must be either "distance" or "rotation" or "combined"')

        branch_input = Input(shape=shape, name='img_input')
        if model == 'distance':
            branch_output = GatePoseEstimator.buld_distance_branch(branch_input, fine_tune)
        elif model == 'combined':
            branch_output = GatePoseEstimator.build_rotation_dist_branch(branch_input, fine_tune)
        else:
            branch_output = GatePoseEstimator.build_rotation_branch(branch_input, fine_tune)

        model = Model(inputs=branch_input, outputs=branch_output,
                      name='GatePoseEstimator')
        print(model.summary())

        return model

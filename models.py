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

from keras.models import Model
from img_utils import crop_and_pad
from keras.layers import BatchNormalization, Dropout, Activation
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Lambda


class GatePoseEstimator:
    @staticmethod
    def build_rotation_branch(inputs, final_act='relu', chan_dim=-1):
        x = Conv2D(16, (3,3), padding="same")(inputs)
        x = Activation('relu')(x)
        x = BatchNormalization(axis=chan_dim)(x)
        x = MaxPooling2D(pool_size=(3, 3))(x)
        x = Dropout(0.25)(x)

        x = Conv2D(32, (3,3), padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization(axis=chan_dim)(x)
        x = Conv2D(64, (3,3), padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization(axis=chan_dim)(x)
        x = MaxPooling2D(pool_size=(2,2))(x)
        x = Dropout(0.25)(x)

        x = Flatten()(x)
        x = Dense(256)(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = Dense(16)(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(1)(x)
        x = Activation(final_act, name='rotation_output')(x)

        return x

    '''
    Use only the bounding box coordinates as input for an MLP for the
    distance estimation.
    '''
    @staticmethod
    def build_distance_branch(inputs, final_act='relu', chan_dim=-1):
        x = Dense(64, activation='relu')(inputs)
        x = Dense(32, activation='relu')(x)
        x = Dense(16, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(1)(x)
        x = Activation(final_act, name='distance_output')(x)

        return x

    @staticmethod
    def build(model, shape, final_act='relu'):
        if model not in ['distance', 'rotation']:
            raise ValueError('Model must be either "distance" or "rotation"')
        if model == 'distance':
            branch_input = Input(shape=(4,), name='bbox_input')
            branch_output = GatePoseEstimator.build_distance_branch(branch_input)
        else:
            branch_input = Input(shape=shape, name='img_input')
            branch_output = GatePoseEstimator.build_rotation_branch(branch_input)

        model = Model(inputs=branch_input, outputs=branch_output,
                      name='GatePoseEstimator')
        print(model.summary())

        return model

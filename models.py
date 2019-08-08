#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 theo <theo@not-arch-linux>
#
# Distributed under terms of the MIT license.

"""
CNN and MLP models
"""

from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D


def CNN(config):
    inputs = Input(shape=config['input_shape'])

    x = Conv2D(64, (3,3), activation='relu', padding='same')(inputs)
    x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2))(x)

    distance_prediction = Dense(1, activation='relu')(x)
    rotation_prediction = Dense(1, activation='relu')(x)

    model = Model(inputs=inputs, outputs=[distance_prediction,
                                          rotation_prediction])
    print(model.summary())

    return model


def MLP(config):
    inputs = Input(shape=config['input_shape'])

    x = Dense(64, activation='relu')(inputs)
    x = Dense(64, activation='relu')(x)

    distance_prediction = Dense(1, activation='relu')(x)
    rotation_prediction = Dense(1, activation='relu')(x)

    model = Model(inputs=inputs, outputs=[distance_prediction,
                                          rotation_prediction])
    print(model.summary())

    return model

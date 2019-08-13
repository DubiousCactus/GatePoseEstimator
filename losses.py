#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 theo <theo@not-arch-linux>
#
# Distributed under terms of the MIT license.

"""
Combined loss
"""

from keras.losses import mean_squared_error

def loss(y_true, y_pred, alpha, beta):
    return (alpha * mean_squared_error(y_true[0:1], y_pred[0:1]) +
            beta * mean_squared_error(y_true[1::], y_pred[1::]))

def combined_loss(alpha, beta):
    def _loss(y_true, y_pred):
        return loss(y_true, y_pred, alpha, beta)
    return _loss

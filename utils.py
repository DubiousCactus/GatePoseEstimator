#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 theo <theo@not-arch-linux>
#
# Distributed under terms of the MIT license.

"""
Different utility functions used for training and testing.
"""

import os
import json
import numpy as np

from keras import backend as K
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, Iterator


class GateGenerator(ImageDataGenerator):
    '''
    Generates batches of images and labels with on-the-go augmentation.
    Extens the ImageDataGenerator from Keras, while only overriding
    flow_from_directory().
    '''
    def flow_from_directory(self, directory, target_size=(225, 300, 1),
                            batch_size=32, shuffle=True,
                            ground_truth_available=False):
        return GateDirectoryIterator(self, directory=directory,
                                     target_size=target_size,
                                     batch_size=batch_size, shuffle=shuffle,
                                     ground_truth_available=ground_truth_available)


class GateDirectoryIterator(Iterator):
    '''
    Parses a directory of images and labels, assuming the following folder
    structure:

        root_folder/
            dataset_1/
                images/
                annotations.json
            dataset_2/
                images/
                annotations.json
            ...

    '''
    def __init__(self, image_data_generator, directory, target_size,
                 batch_size, shuffle, ground_truth_available=False):
        self.image_data_generator = image_data_generator
        self.directory = directory
        self.target_size = tuple(target_size)
        self.batch_size = batch_size
        self.shuffle = shuffle
        if len(target_size) < 3 and target_size[2] not in [1, 3]:
            raise ValueError('Invalid target size: ', target_size,
                             '. Excpeted (H, W, C).')
        self.samples = 0
        self.images_dir = []
        self.annotations_filenames = []
        self.filenames = []
        self.labels = []
        self.color_mode = 'rgb' if target_size[2] == 3 else 'grayscale'
        if not ground_truth_available:
            self.labels = None

        if not os.path.exists(self.directory):
            raise ValueError("Directory {} does not exist.".format(self.directory))

        for root, subdirs, files in os.walk(self.directory):
            if 'images' in subdirs and 'annotations.json' in files:
                self.images_dir.append(os.path.join(self.directory, root, 'images'))
                self.annotations_filenames.append(os.path.join(self.directory,
                                                               root,
                                                               'annotations.json'))

        with open(self.annotations_filenames[0], 'r') as f:
            annotations = json.load(f)

        for images_dir, annotations_filename in zip(self.images_dir,
                                                    self.annotations_filenames):
            with open(annotations_filename) as f:
                annotations = json.load(f)

            for i, img_annotation in enumerate(annotations['annotations']):
                img = img_annotation['image']
                annotations = img_annotation['annotations']
                for bbox in annotations:
                    if bbox['class_id'] in [1, 2]:
                        self.filenames.append(os.path.join(images_dir, img))
                        if ground_truth_available:
                            self.labels.append(bbox)

        self.samples = len(self.filenames)

        assert self.samples > 0, "Empty dataset!"

        print("Loaded {} images".format(self.samples))

        super(GateDirectoryIterator, self).__init__(self.samples,
                                                    batch_size, shuffle,
                                                    seed=None)


    def next(self):
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock so it can
        # be done in parallel
        return self._get_batches_of_transformed_samples(index_array)


    def _get_batches_of_transformed_samples(self, index_array):
        '''
        Fetches the next batch of images and labels, and applies
        transformations.
        '''
        current_batch_size = index_array.shape[0]
        batch_x1 = np.empty((current_batch_size,) + self.target_size,
                           dtype=K.floatx())
        batch_x2 = np.empty((current_batch_size, 4,), dtype=K.floatx())
        batch_dist = np.empty((current_batch_size, 1,), dtype=K.floatx())
        batch_rot = np.empty((current_batch_size, 1,), dtype=K.floatx())

        for i, j in enumerate(index_array):
            x = image.load_img(os.path.join(self.directory, self.filenames[j]),
                              target_size=self.target_size,
                               color_mode=self.color_mode)
            x = image.img_to_array(x)
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)

            batch_x1[i] = x
            batch_x2[i] = np.array([
                self.labels[index_array[i]]['xmin'],
                self.labels[index_array[i]]['ymin'],
                self.labels[index_array[i]]['xmax'],
                self.labels[index_array[i]]['ymax'],
            ])
            batch_dist[i] = self.labels[index_array[i]]['distance']
            batch_rot[i] = self.labels[index_array[i]]['rotation']

        return ({'img_input': batch_x1, 'bbox_input': batch_x2},
               {'distance_output': batch_dist, 'rotation_output': batch_rot})

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

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, Iterator


class GateGenerator(ImageDataGenerator):
    '''
    Generates batches of images and labels with on-the-go augmentation.
    Extens the ImageDataGenerator from Keras, while only overriding
    flow_from_directory().
    '''
    def flow_from_directory(self, directory, target_size=(225, 300, 1),
                            batch_size=32, shuffle=True):
        return GateDirectoryIterator(self, directory=directory,
                                     target_size=target_size,
                                     batch_size=batch_size, shuffle=shuffle)


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
                 batch_size, shuffle, ground_truth_available=True):
        self.image_data_generator = image_data_generator
        self.directory = directory
        self.target_size = target_size
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
        if not ground_truth_available:
            self.labels = None

        for root, subdirs, files in os.walk(datasets_dir):
            if 'images' in subdirs and 'annotations.json' in files:
                self.images_dir.append(os.path.join(datasets_dir, root, 'images'))
                self.annotations_filenames.append(os.path.join(datasets_dir,
                                                               root,
                                                               'annotations.json'))

        with open(self.annotations_filenames[0], 'r') as f:
            annotations = json.load(f)

        self.classes_to_labels = annotations['classes']
        for images_dir, annotations_filename in zip(self.images_dir,
                                                    self.annotations_filenames):
            with open(annotations_filename) as f:
                annotations = json.load(f)

            for i, img_annotations in enumerate(annotations['annotations']):
                img = img_annotations['image']
                annotations = img_annotations['annotations']
                self.filenames.append(os.path.join(images_dir, img))

                if ground_truth_available:
                    self.labels.append(annotations)

        self.dataset_size = len(self.filenames)
        self.labels = np.array(self.labels, dtype=K.floatx())

        print("Loaded {} images".format(self.dataset_size))

        super(GateDirectoryIterator, self).__init__(self.dataset_size,
                                                    batch_size, shuffle)


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
        batch_x = np.zeros((current_batch_size,) + self.image_shape,
                           dtype=K.floatx())
        batch_dist = np.zeros((current_batch_size, 2, ), dtype=K.floatx())
        batch_rot = np.zeros((current_batch_size, 2, ), dtype=K.floatx())

        for i, j in enumerate(index_array):
            x = image.load_img(os.path.join(self.directory, self.filenames[j]),
                              target_size=self.target_size)
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x

            batch_dist[i,0] = 0.0
            batch_dist[i,1] = self.labels[index_array[i]]['distance']
            batch_rot[i,0] = 1.0
            batch_rot[i,1] = self.labels[index_array[i]]['rotation']

        return batch_x, [batch_dist, batch_rot]

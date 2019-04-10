# Copyright 2019 Babylon Partners. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import numpy as np
import tensorflow as tf

from tensorflow.contrib.learn import ModeKeys

from rgat.datasets import rdf


def sp2tfsp(x):
    coo = x.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)


def get_input_fn(
        mode,
        dataset_name,
        validation=True,
        name='data'):
    """Build the input function from RDF dataset.

    Args:
        mode (str): The current modality, one of 'train', 'eval', 'infer'.
        dataset_name (str): Specifies type of the RDF Dataset
        name (str): The name of the data set for variable name scoping. Defaults
            to 'data'.
        validation (bool): Whether to do validation. Defaults to `True`.

    Returns:
        tuple(dict, dict) The dictionaries corresponding to the values for x and
            y provided by the generator.
    """
    ModeKeys.validate(mode)

    data_dict = rdf.get_dataset(dataset_name)

    # Convert to SparseTensors
    support = {k: sp2tfsp(v) for (k, v) in data_dict['support'].items()}
    features = sp2tfsp(data_dict['features'])

    y_train, y_val, y_test, idx_train, idx_val, idx_test = get_splits(
        y=data_dict['labels'],
        train_idx=data_dict['train_idx'],
        test_idx=data_dict['test_idx'],
        validation=validation)

    if mode == ModeKeys.TRAIN:
        y, y_ind = y_train, idx_train
    elif mode == ModeKeys.EVAL:
        y, y_ind = y_val, idx_val
    else:
        y, y_ind = y_test, idx_test

    # Convert y to an integer representation
    y = np.argmax(y, axis=1)

    def input_fn():
        with tf.name_scope(name):
            dataset = tf.data.Dataset.from_tensors(
                {'labels': y,
                 'support': support,
                 'mask': y_ind})

            dataset = dataset.repeat()

            iterator = dataset.make_one_shot_iterator()

            next_elements = iterator.get_next()

            next_features = {
                'features': features, 'support': next_elements['support']}
            next_labels = {k: next_elements[k] for k in ['labels', 'mask']}

            return next_features, next_labels

    return input_fn


def get_splits(y, train_idx, test_idx, validation):
    if validation:
        tf.logging.info("Training on 80% of training set, evaluating on 20% of "
                        "training set. Test set is the test set, do not use "
                        "it.")
        idx_train = train_idx[int(len(train_idx) / 5):]
        idx_val = train_idx[:int(len(train_idx) / 5)]
        idx_test = test_idx
    else:
        tf.logging.info("Training on training set, evaluating on "
                        "training set. Test set is the test set, use at your "
                        "peril.")
        idx_train = train_idx
        idx_val = train_idx  # NB not not validation
        idx_test = test_idx

    tf.logging.info("Train set size: {}".format(len(idx_train)))
    tf.logging.info("Validation set size: {}".format(len(idx_val)))
    tf.logging.info("Test set size: {}".format(len(idx_test)))

    y_train = np.zeros(y.shape)
    y_val = np.zeros(y.shape)
    y_test = np.zeros(y.shape)

    y_train[idx_train] = np.array(y[idx_train].todense())
    y_val[idx_val] = np.array(y[idx_val].todense())
    y_test[idx_test] = np.array(y[idx_test].todense())

    return y_train, y_val, y_test, idx_train, idx_val, idx_test

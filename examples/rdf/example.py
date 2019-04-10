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
"""AIFB and MUTAG semi-supervised classification task example.
"""
import os

import tensorflow as tf

from tensorflow.contrib.learn import ModeKeys

from rgat.datasets import rdf
from rgat.layers import RGAT, RGC

from inputs import get_input_fn

tf.logging.set_verbosity(tf.logging.DEBUG)

FLAGS = tf.flags.FLAGS

# Run config
tf.flags.DEFINE_string("base_dir", "/tmp/runs",
                       "The base path for experiments.")
tf.flags.DEFINE_integer("save_summary_steps", 10,
                        "The frequency to save summaries in steps.")
tf.flags.DEFINE_integer("save_checkpoints_steps", 10,
                        "The frequency to save checkpoints in steps.")
tf.flags.DEFINE_integer("tf_random_seed", 1234, "The random seed.")

# Hyperparameters

# Dataset and training
tf.flags.DEFINE_string("dataset", "AIFB",
                       "The dataset. One of `'AIFB'` or `'MUTAG'`.")
tf.flags.DEFINE_integer("max_steps", 50, "The number of training steps.")
tf.flags.DEFINE_float("learning_rate", 1.e-2, "The learning rate.")

# Layer
tf.flags.DEFINE_string("model", "rgat",
                       "The model to run. One of `'rgat'` and `'rgc'`. "
                       "NB, `'rgc'` is much faster.")
tf.flags.DEFINE_integer("units", 16,
                        "The number of units in the first layer.")
tf.flags.DEFINE_integer("attention_units", 1,
                        "The number of units in the first attention layer. This"
                        " should be `1` if we set attention style to `'sum'`")
tf.flags.DEFINE_string("head_aggregation", "concat",
                       "The head aggregation style in the first layer. One of "
                       "`'concat'` or `'mean'`.")
tf.flags.DEFINE_integer("heads", 4,
                        "The number of attention heads in the first layer.")
tf.flags.DEFINE_string("attention_style", "sum",
                       "The attention style. One of `'sum'` or `'dot'`.")
tf.flags.DEFINE_string("attention_mode", "argat",
                       "The attention mode. One of `'argat'` or `'wirgat'`.")


def get_relations_classes(dataset):
    dataset = rdf.get_dataset(dataset)
    return len(dataset['support']), dataset['labels'].shape[-1]


class RGATNModel(tf.keras.Model):
    def __init__(self, params, training):
        super(RGATNModel, self).__init__()
        self.units = params.units
        self.heads = params.heads
        self.relations = params.relations
        self.head_aggregation = params.head_aggregation
        self.attention_mode = params.attention_mode
        self.attention_style = params.attention_style
        self.attention_units = params.attention_units

        self.rgat1 = RGAT(units=self.units[0],
                          relations=self.relations,
                          heads=self.heads,
                          head_aggregation=self.head_aggregation,
                          attention_mode=self.attention_mode,
                          attention_style=self.attention_style,
                          attention_units=self.attention_units,
                          activation=tf.nn.relu)
        self.rgat2 = RGAT(units=self.units[1],
                          relations=self.relations,
                          attention_mode=self.attention_mode,
                          attention_style=self.attention_style,
                          attention_units=self.attention_units)

    def call(self, inputs, support):
        x = self.rgat1(inputs=inputs, support=support)
        return self.rgat2(inputs=x, support=support)


class RGCNModel(tf.keras.Model):
    def __init__(self, params, training):
        super(RGCNModel, self).__init__()
        self.units = params.units
        self.relations = params.relations

        self.rgc1 = RGC(units=self.units[0],
                        relations=self.relations,
                        activation=tf.nn.relu)
        self.rgc2 = RGC(units=self.units[1],
                        relations=self.relations)

    def call(self, inputs, support):
        x = self.rgc1(inputs=inputs, support=support)
        return self.rgc2(inputs=x, support=support)


def model_fn(features, labels, mode, params):
    training = mode == ModeKeys.TRAIN

    if params.model == "rgat":
        model_class = RGATNModel
    elif params.model == "rgc":
        model_class = RGCNModel
    else:
        raise ValueError(
            "Unknown model {}. Must be one of `'rgat'` or `'rgc'`".format(
                params.model))

    model = model_class(params=params, training=training)

    inputs, support = features['features'], features['support']

    # Combine dict of supports into a single matrix
    support = tf.sparse_concat(axis=1, sp_inputs=list(support.values()),
                               name="combine_supports")

    logits = model(inputs=inputs, support=support)

    predictions = tf.argmax(logits, axis=-1, name='predictions')

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode, predictions={'logits': logits, 'predictions': predictions})

    mask, labels = labels['mask'], labels['labels']

    # Get only unmasked labels, logits and predictions
    labels, logits = tf.gather(labels, mask), tf.gather(logits, mask)
    predictions = tf.gather(predictions, mask)

    loss = tf.losses.sparse_softmax_cross_entropy(
        labels=labels, logits=logits)

    with tf.name_scope('metrics'):
        accuracy = tf.metrics.accuracy(
            labels=labels, predictions=predictions)

    metrics = {'accuracy': accuracy}

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=metrics)

    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate)
    global_step = tf.train.get_global_step()
    train_op = optimizer.minimize(loss, global_step=global_step)
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def main(unused_argv):
    tf.logging.info("{} Flags {}".format('*'*15, '*'*15))
    for k, v in FLAGS.flag_values_dict().items():
        tf.logging.info("FLAG `{}`: {}".format(k, v))
    tf.logging.info('*' * (2 * 15 + len(' Flags ')))

    if FLAGS.dataset not in rdf.ALLOWED_DATASETS:
        raise ValueError("Unrecognised dataset {}. Must be one of {}.".format(
            FLAGS.dataset, rdf.ALLOWED_DATASETS))

    relations, classes = get_relations_classes(dataset=FLAGS.dataset)

    hparams = tf.contrib.training.HParams(
        model=FLAGS.model,
        relations=relations,
        units=[FLAGS.units, classes],
        attention_units=FLAGS.attention_units,
        head_aggregation=FLAGS.head_aggregation,
        heads=FLAGS.heads,
        attention_style=FLAGS.attention_style,
        attention_mode=FLAGS.attention_mode,
        max_steps=FLAGS.max_steps,
        learning_rate=FLAGS.learning_rate)

    config = tf.estimator.RunConfig(
        tf_random_seed=FLAGS.tf_random_seed,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        save_summary_steps=FLAGS.save_summary_steps)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=os.path.join(FLAGS.base_dir, FLAGS.dataset, FLAGS.model),
        config=config,
        params=hparams)

    train_input_fn = get_input_fn(
        mode=ModeKeys.TRAIN, dataset_name=FLAGS.dataset)
    eval_input_fn = get_input_fn(
        mode=ModeKeys.EVAL, dataset_name=FLAGS.dataset)

    tf.estimator.train_and_evaluate(
        estimator,
        train_spec=tf.estimator.TrainSpec(
            train_input_fn, max_steps=hparams.max_steps),
        eval_spec=tf.estimator.EvalSpec(
            eval_input_fn, steps=1, throttle_secs=0))


if __name__ == '__main__':
    tf.app.run(main=main)

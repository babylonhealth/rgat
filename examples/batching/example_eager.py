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
"""Eager mode batching example.
"""
import numpy as np
import tensorflow as tf

from collections import OrderedDict
from scipy import sparse

from rgat.layers import RGAT
from rgat.utils import graph_utils

tf.enable_eager_execution()
tf.logging.set_verbosity(tf.logging.DEBUG)

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer("seed", 42, "The random seed.")
tf.flags.DEFINE_integer("relations", 3, "The number of relations.")
tf.flags.DEFINE_integer("nodes_min", 2,
                        "The smallest number of nodes any graph can have. This "
                        "is used for random graph generation.")
tf.flags.DEFINE_integer("nodes_max", 9,
                        "The largest number of nodes any graph can have. This "
                        "is used for random graph generation.")
tf.flags.DEFINE_integer("batch_size", 37, "The batch size.")
tf.flags.DEFINE_integer("attention_heads", 7, "The number of attention heads.")
tf.flags.DEFINE_integer("features_dim", 3, "The input dimensionality.")
tf.flags.DEFINE_integer("units", 5, "The number of units in the layer.")


def _build_support(size):
    sup = tf.random.uniform(shape=(size, size))
    sup = sup > 0.75
    sup = tf.cast(sup, tf.int32)
    return sparse.coo_matrix(sup)


def _built_relational_support(size, names):
    return OrderedDict([(r, _build_support(size)) for r in names])


def get_batch_of_features_supports_values():
    tf.logging.info("Generating support names.")
    rel_names = ["rel_{}".format(i) for i in range(FLAGS.relations)]

    tf.logging.info("Generating number of nodes in each element of the batch.")
    graph_sizes = [
        tf.random.uniform(
            shape=[], minval=FLAGS.nodes_min, maxval=FLAGS.nodes_max,
            dtype=tf.int32)
        for _ in range(FLAGS.batch_size)]

    tf.logging.info("Generating fake input features for each node in each "
                    "graph.")
    features = [tf.random.uniform(shape=(graph_size, FLAGS.features_dim))
                for graph_size in graph_sizes]

    supports = [
        _built_relational_support(size=graph_size, names=rel_names)
        for graph_size in graph_sizes]

    return features, supports


def main(unused_argv):
    tf.logging.info("{} Flags {}".format('*'*15, '*'*15))
    for k, v in FLAGS.flag_values_dict().items():
        tf.logging.info("FLAG `{}`: {}".format(k, v))
    tf.logging.info('*' * (2 * 15 + len(' Flags ')))

    np.random.seed(FLAGS.seed)
    tf.set_random_seed(FLAGS.seed)

    rgat_layer = RGAT(units=FLAGS.units, relations=FLAGS.relations)

    features, supports = get_batch_of_features_supports_values()

    # Route 1: Run RGAT on each element in the batch separately and combine the
    # results
    individual_supports = [
        graph_utils.relational_supports_to_support(d) for d in supports]
    individual_supports = [
        graph_utils.triple_from_coo(s) for s in individual_supports]
    individual_supports = [
        tf.SparseTensor(i, v, ds) for i, v, ds in individual_supports]
    individual_supports = [
        tf.sparse_reorder(s) for s in individual_supports]

    individual_results = [
        rgat_layer(inputs=f, support=s)
        for f, s in zip(features, individual_supports)]
    individual_results = tf.concat(individual_results, axis=0)

    # Route 2: First combine the batch into a single graph and pass everything
    # through in one go
    combined_features = tf.concat(features, axis=0)

    combined_supports = graph_utils.batch_of_relational_supports_to_support(
        supports)
    combined_supports = graph_utils.triple_from_coo(combined_supports)
    combined_supports = tf.SparseTensor(*combined_supports)
    combined_supports = tf.sparse_reorder(combined_supports)

    combined_results = rgat_layer(
        inputs=combined_features, support=combined_supports)

    if np.allclose(combined_results, individual_results):
        tf.logging.info("The approaches match!")
    else:
        raise ValueError(
            "Doing each element in a batch independently does not produce the "
            "same results as doing all the batch in one go. Something has "
            "clearly broken. Please contact the author ASAP :).")


if __name__ == '__main__':
    tf.app.run(main=main)

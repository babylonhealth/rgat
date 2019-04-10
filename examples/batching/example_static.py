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
"""Static graph batching example.
"""
import numpy as np
import tensorflow as tf

from collections import OrderedDict
from scipy import sparse

from rgat.layers import RGAT
from rgat.utils import graph_utils

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
    sup = np.random.uniform(size=(size, size))
    sup = (sup > 0.75).astype(int)
    return sparse.coo_matrix(sup)


def _built_relational_support(size, names):
    return OrderedDict([(r, _build_support(size)) for r in names])


def get_architecture():
    inputs_ph = tf.placeholder(
        dtype=tf.float32, shape=[None, FLAGS.features_dim], name="features_")
    support_ph = tf.sparse_placeholder(
        dtype=tf.float32, shape=[None, None], name="support_")

    tf.logging.info("Reordering indices of support - this is extremely "
                    "important as sparse operations assume sparse indices have "
                    "been ordered.")
    support_reorder = tf.sparse_reorder(support_ph)

    rgat_layer = RGAT(units=FLAGS.units, relations=FLAGS.relations)

    outputs = rgat_layer(inputs=inputs_ph, support=support_reorder)

    return inputs_ph, support_ph, outputs


def get_batch_of_features_supports_values():
    tf.logging.info("Generating support names.")
    rel_names = ["rel_{}".format(i) for i in range(FLAGS.relations)]

    tf.logging.info("Generating number of nodes in each element of the batch.")
    graph_sizes = [
        np.random.random_integers(low=FLAGS.nodes_min, high=FLAGS.nodes_max)
        for _ in range(FLAGS.batch_size)]

    tf.logging.info("Generating fake input features for each node in each "
                    "graph.")
    features_val = [np.random.uniform(size=(graph_size, FLAGS.features_dim))
                    for graph_size in graph_sizes]

    supports_val = [
        _built_relational_support(size=graph_size, names=rel_names)
        for graph_size in graph_sizes]

    return features_val, supports_val


def main(unused_argv):
    tf.logging.info("{} Flags {}".format('*'*15, '*'*15))
    for k, v in FLAGS.flag_values_dict().items():
        tf.logging.info("FLAG `{}`: {}".format(k, v))
    tf.logging.info('*' * (2 * 15 + len(' Flags ')))

    np.random.seed(FLAGS.seed)
    tf.set_random_seed(FLAGS.seed)

    features_ph, support_ph, outputs = get_architecture()

    features_val, supports_val = get_batch_of_features_supports_values()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Route 1: Run RGAT on each element in the batch separately and combine the
    # results
    individual_supports = [
        graph_utils.relational_supports_to_support(d) for d in supports_val]
    individual_supports = [
        graph_utils.triple_from_coo(s) for s in individual_supports]

    individual_results = [
        sess.run(outputs, feed_dict={features_ph: fv, support_ph: sv})
        for fv, sv in zip(features_val, individual_supports)]
    individual_results = np.concatenate(individual_results, axis=0)

    # Route 2: First combine the batch into a single graph and pass everything
    # through in one go
    combined_features_val = np.concatenate(features_val, axis=0)
    combined_supports = graph_utils.batch_of_relational_supports_to_support(
        supports_val)
    combined_supports = graph_utils.triple_from_coo(combined_supports)

    combined_results = sess.run(
        outputs, feed_dict={features_ph: combined_features_val,
                            support_ph: combined_supports})

    if np.allclose(combined_results, individual_results):
        tf.logging.info("The approaches match!")
    else:
        raise ValueError(
            "Doing each element in a batch independently does not produce the "
            "same results as doing all the batch in one go. Something has "
            "clearly broken. Please contact the author ASAP :).")


if __name__ == '__main__':
    tf.app.run(main=main)

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
"""Attention layer for generating graph attention logits.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.keras import activations
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import layers as keras_layers
from tensorflow.python.keras.engine import InputSpec

from rgat import layers as rgat_layers
from rgat.ops import math_ops as rgat_math_ops
from rgat.ops import sparse_ops as rgat_sparse_ops
from rgat.layers.graph_utils import AttentionStyles


class RelationalGraphAttentionLogits(keras_layers.Layer):
    """Layer implementing generalisation of attention logits of
    https://arxiv.org/abs/1710.10903.

    """
    def __init__(self,
                 relations,
                 heads=1,
                 basis_size=None,
                 activation=tf.nn.leaky_relu,
                 attention_style=AttentionStyles.SUM,
                 attention_units=1,
                 use_edge_features=False,
                 kernel_initializer='glorot_uniform',
                 edge_kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 edge_kernel_regularizer=None,
                 activity_regularizer=None,
                 feature_dropout=None,
                 edge_feature_dropout=None,
                 batch_normalisation=False,
                 name='graph_attention_logits',
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(RelationalGraphAttentionLogits, self).__init__(
            activity_regularizer=regularizers.get(activity_regularizer),
            name=name, **kwargs)

        self.heads = int(heads)
        self.relations = int(relations)
        self.basis_size = int(basis_size) if basis_size else None
        self.activation = activations.get(activation)
        self.attention_style = AttentionStyles.validate(attention_style)
        if self.attention_style == AttentionStyles.SUM:
            assert attention_units == 1, (
                "If your attention style is {} then you should only have 1 "
                "attention_units, you have {}".format(AttentionStyles.SUM,
                                                      attention_units))
        self.attention_units = attention_units  # g
        self.use_edge_features = use_edge_features
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.edge_kernel_initializer = initializers.get(edge_kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.edge_kernel_regularizer = regularizers.get(edge_kernel_regularizer)
        self.feature_dropout = feature_dropout
        self.edge_feature_dropout = edge_feature_dropout
        self.batch_normalisation = batch_normalisation
        self.supports_masking = True
        self.input_spec = InputSpec(min_ndim=2)

        self.dense_layer = rgat_layers.BasisDecompositionDense(
            units=self.heads * self.relations * self.attention_units * 2,
            basis_size=self.basis_size,
            coefficients_size=self.relations * self.heads,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name=name + '_basis_decomposition_dense',
            **kwargs)
        if self.use_edge_features:
            self.edges_dense_layer = tf.keras.layers.Dense(
                units=self.heads,
                use_bias=False,
                kernel_initializer=self.edge_kernel_initializer,
                kernel_regularizer=self.edge_kernel_regularizer,
                name=name + '_edge_dense',
                **kwargs)
        if self.batch_normalisation:
            self.batch_normalisation_layer = tf.layers.BatchNormalization()

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        if input_shape[-1].value is None:
            raise ValueError('The last dimension of the inputs to '
                             '`GraphAttentionLogits` should be defined. '
                             'Found `None`.')
        self.input_spec = InputSpec(min_ndim=2,
                                    axes={-1: input_shape[-1].value})

        self.input_units = input_shape[-1].value / (self.relations * self.heads)
        assert self.input_units.is_integer()
        self.input_units = int(self.input_units)

        fake_input_shape = tf.TensorShape([1, self.input_units])
        self.dense_layer.build(input_shape=fake_input_shape)

        self.kernel = self.dense_layer.kernel
        self.kernel = tf.reshape(
            self.kernel, [self.relations * self.heads, self.input_units,
                          self.attention_units * 2])

        self.built = True

    def call(self, inputs, support, edge_features=None, as_sparse_tensor=False,
             training=False):
        if self.use_edge_features:
            assert edge_features is not None, (
                "You must provide edge_features "
                "because self.use_edge_features is `True`.")
            edge_features = tf.convert_to_tensor(edge_features,
                                                 dtype=self.dtype)       # E,G
        else:
            assert edge_features is None, (
                "You can not provide edge_features "
                "because self.use_edge_features is `False`.")

        inputs = tf.convert_to_tensor(inputs, dtype=self.dtype)          # N,RHF

        # support is N,RN

        self._nodes = n = rgat_math_ops.get_shape(inputs)[0]
        f, h, r, g = (self.input_units, self.heads, self.relations,
                      self.attention_units)

        if self.feature_dropout is not None:
            inputs = tf.nn.dropout(inputs, keep_prob=1 - self.feature_dropout,
                                   name="feature_dropout")               # N,RHF

        with tf.name_scope("transform_rh_n_f"):
            inputs_n_rh_f = tf.reshape(inputs, [n, r * h, f])
            inputs_rh_n_f = tf.transpose(inputs_n_rh_f, [1, 0, 2])

        # self.kernel is RH,F,G (G=1 for SUM style attention)
        logits_rh_n_2g = tf.matmul(inputs_rh_n_f, self.kernel)

        with tf.name_scope("transform_rn_h_2g"):
            logits_r_h_n_2g = tf.reshape(logits_rh_n_2g, [r, h, n, 2 * g])
            logits_r_n_h_2g = tf.transpose(logits_r_h_n_2g, [0, 2, 1, 3])
            logits_rn_h_2g = tf.reshape(logits_r_n_h_2g, [r * n, h, 2 * g])

        query_rn_h_g, key_rn_h_g = tf.split(
            logits_rn_h_2g, num_or_size_splits=2, axis=-1,
            name="split_query_key")                                 # 2x[RN,H,G]

        support_indices_to, support_indices_from = tf.split(
            support.indices, num_or_size_splits=2, axis=-1,
            name="split_indices_to_from")                              # 2x[E,1]

        with tf.name_scope("gather_edge_representations"):
            query_e_h_g = tf.gather_nd(query_rn_h_g,
                                       indices=support_indices_to,
                                       name="query_edges")
            key_e_h_g = tf.gather_nd(key_rn_h_g,
                                     indices=support_indices_from,
                                     name="key_edges")

        with tf.name_scope("logits_{}_e_h".format(self.attention_style)):
            if self.attention_style == AttentionStyles.SUM:
                query_e_h = tf.squeeze(query_e_h_g, axis=-1)
                key_e_h = tf.squeeze(key_e_h_g, axis=-1)
                logits_e_h = tf.add(query_e_h, key_e_h)                    # E,H

            elif self.attention_style == AttentionStyles.DOT:
                query_e_h_1_g = tf.reshape(query_e_h_g, [-1, h, 1, g])
                key_e_h_g_1 = tf.reshape(key_e_h_g, [-1, h, g, 1])
                logits_e_h_1_1 = tf.matmul(query_e_h_1_g, key_e_h_g_1)
                logits_e_h = tf.squeeze(logits_e_h_1_1, [-1, -2])

        if self.use_edge_features:
            if self.edge_feature_dropout is not None:
                edge_features = tf.nn.dropout(
                    edge_features, keep_prob=1 - self.edge_feature_dropout,
                    name="edge_feature_dropout")  # E,G
            edge_features_logits_e_h = self.edges_dense_layer(edge_features)
            logits_e_h = tf.add(logits_e_h, edge_features_logits_e_h)

        if self.batch_normalisation:
            logits_e_h = self.batch_normalisation_layer(logits_e_h,
                                                        training=training)

        if self.activation is not None:
            logits_e_h = self.activation(logits_e_h)  # pylint: disable=not-callable

        if not as_sparse_tensor:
            return logits_e_h

        input_ind = support.indices                                     # E,2
        input_shape = support.dense_shape                               # N,RN

        output_ind = rgat_sparse_ops.indices_expand_0(
            indices=input_ind, zeroth_dim=self.heads)               # HE,3

        logits_h_e = tf.transpose(logits_e_h, perm=[1, 0])
        output_val = tf.reshape(logits_h_e, [-1])                   # HE
        output_shape = tf.concat([[self.heads], input_shape],
                                 axis=-1)                           # H,RN,N

        return tf.SparseTensor(output_ind, output_val, output_shape)

    def compute_output_shape(self, input_shape):
        return self.dense_layer_1.compute_output_shape(input_shape)

    def get_config(self):
        config = {
            'relations': self.relations,
            'heads': self.heads,
            'basis_size': self.basis_size,
            'activation': activations.serialize(self.activation),
            'attention_style': self.attention_style,
            'attention_features': self.attention_units,
            'use_edge_features': self.use_edge_features,
            'kernel_initializer': initializers.serialize(
                self.kernel_initializer),
            'edge_kernel_initializer': initializers.serialize(
                self.edge_kernel_initializer),
            'edge_kernel_regularizer': regularizers.serialize(
                self.edge_kernel_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'feature_dropout': self.feature_dropout,
            'edge_feature_dropout': self.feature_dropout,
            'batch_normalisation': self.batch_normalisation
        }
        base_config = super(RelationalGraphAttentionLogits, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

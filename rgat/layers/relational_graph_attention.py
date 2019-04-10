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
"""Relational graph attention layer.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import tensor_shape

from tensorflow.python.keras import activations
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import layers as keras_layers
from tensorflow.python.keras.engine import InputSpec

from rgat import layers as rgat_layers
from rgat.ops import math_ops as rgat_math_ops
from rgat.layers.graph_utils import HeadAggregation
from rgat.layers.graph_utils import AttentionModes
from rgat.layers.graph_utils import AttentionStyles

from .relational_graph_attention_logits import RelationalGraphAttentionLogits


class RelationalGraphAttention(keras_layers.Layer):
    """Layer implementing Relational Graph Attention of
    https://openreview.net/forum?id=Bklzkh0qFm with sparse supports.

    Must be called using both inputs and support:
        inputs = get_inputs()
        support = get_support()

        rgat_layer = RelationalGraphAttention(...)
        outputs = rgat_layer(inputs=inputs, support=support)

    Has alias of `RGAT`.

    Arguments:
        units (int): The dimensionality of the output space.
        relations (int): The number of relation types the layer will handle.
        heads (int): The number of attention heads to use (see
            https://arxiv.org/abs/1710.10903). Defaults to `1`.
        head_aggregation (str): The attention head aggregation method to use
            (see https://arxiv.org/abs/1710.10903). Can be one of `'mean'` or
            `'concat'`. Defaults to `'mean'`.
        attention_mode (str): The relational attention mode to to use (see
            https://openreview.net/forum?id=Bklzkh0qFm). Can be one of `'argat'`
            or `'wirgat'`. Defaults to `'argat'`.
        attention_style (str): The different types of attention to use. To use
            the transformer style multiplicative attention, set to `'dot'`.  To
            use the GAT style additive attention set to `'sum'`. Defaults to
            `'sum'`.
        attention_units (int): The dimensionality of the attention space. If
            using `'sum'` style attention, this must be set to `1`.
        attn_use_edge_features (bool): Whether the layer can use edge features.
            Defaults to `False`.
        kernel_basis_size (int): The number of basis kernels to create the
            relational kernels from, i.e. W_r = sum_i c_{i,r} W'_i, where
            r = 1, 2, ..., relations, and i = 1, 2 ..., kernel_basis_size.
            If `None` (default), these is no basis decomposition.
        attn_kernel_basis_size (int): The number of basis kernels to create the
            relational attention kernels from. Defaults to `None`.
        activation (callable): Activation function. Set it to `None` to maintain
            a linear activation.
        attn_activation (callable): Activation function to apply to the
            attention logits prior to feeding to softmax. Defaults to the leaky
            relu in https://arxiv.org/abs/1710.10903, however, when using
            `'dot'` style attention, this can be set to `None`.
        use_bias (bool): Whether the layer uses a bias. Defaults to `False`.
        batch_normalisation (bool): Whether the layer uses batch normalisation.
            Defaults to `False`.
        kernel_initializer (callable): Initializer function for the graph
            convolution weight matrix. If None (default), weights are
            initialized using the `glorot_uniform` initializer.
        bias_initializer (callable): Initializer function for the bias. Defaults
            to `zeros`.
        attn_kernel_initializer (callable): Initializer function for the
            attention weight matrix. If None (default), weights are
            initialized using the `glorot_uniform` initializer.
        kernel_regularizer (callable): Regularizer function for the graph
            convolution weight matrix. Defaults to `None`.
        bias_regularizer (callable): Regularizer function for the bias. Defaults
            to `None`.
        attn_kernel_regularizer (callable): Regularizer function for the graph
            attention weight matrix. Defaults to `None`.
        activity_regularizer (callable): Regularizer function for the output.
            Defaults to `None`.
        feature_dropout (float): The dropout rate for node feature
            representations, between 0 and 1. E.g. rate=0.1 would drop out 10%
            of node input units.
        support_dropout (float): The dropout rate for edges in the support,
            between 0 and 1. E.g. rate=0.1 would drop out 10%
            of the edges in the support.
        edge_feature_dropout (float): The dropout rate for edge feature
            representations, between 0 and 1.
        name (string): The name of the layer. Defaults to
            `rgat`.

    """
    def __init__(self,
                 units,
                 relations,
                 heads=1,
                 head_aggregation=HeadAggregation.MEAN,
                 attention_mode=AttentionModes.ARGAT,
                 attention_style=AttentionStyles.SUM,
                 attention_units=1,
                 attn_use_edge_features=False,
                 kernel_basis_size=None,
                 attn_kernel_basis_size=None,
                 activation=None,
                 attn_activation=tf.nn.leaky_relu,
                 use_bias=False,
                 batch_normalisation=False,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 attn_kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 attn_kernel_regularizer=None,
                 activity_regularizer=None,
                 feature_dropout=None,
                 support_dropout=None,
                 edge_feature_dropout=None,
                 name='rgat',
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(RelationalGraphAttention, self).__init__(
            activity_regularizer=regularizers.get(activity_regularizer),
            name=name, **kwargs)

        self.units = int(units)
        self.relations = int(relations)
        self.heads = int(heads)
        self.head_aggregation = HeadAggregation.validate(head_aggregation)
        self.attention_mode = AttentionModes.validate(attention_mode)
        self.attention_style = AttentionStyles.validate(attention_style)
        self.attention_units = attention_units
        self.attn_use_edge_features = attn_use_edge_features

        self.kernel_basis_size = (int(kernel_basis_size)
                                  if kernel_basis_size else None)
        self.attn_kernel_basis_size = (int(attn_kernel_basis_size)
                                       if attn_kernel_basis_size else None)

        self.activation = activations.get(activation)
        self.attn_activation = activations.get(attn_activation)

        self.use_bias = use_bias
        self.batch_normalisation = batch_normalisation

        if self.batch_normalisation:
            self.batch_normalisation_layer = tf.layers.BatchNormalization()

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.attn_kernel_initializer = initializers.get(attn_kernel_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.attn_kernel_regularizer = regularizers.get(attn_kernel_regularizer)

        self.feature_dropout = feature_dropout
        self.support_dropout = support_dropout
        self.edge_feature_dropout = edge_feature_dropout

        self.supports_masking = True
        self.input_spec = InputSpec(min_ndim=2)

        self.dense_layer = rgat_layers.BasisDecompositionDense(
            units=self.relations * self.heads * self.units,
            basis_size=self.kernel_basis_size,
            coefficients_size=self.relations * self.heads,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name=name + '_basis_decomposition_dense',
            **kwargs)
        self.attention_logits = RelationalGraphAttentionLogits(
            relations=self.relations,
            heads=self.heads,
            attention_style=self.attention_style,
            attention_units=self.attention_units,
            basis_size=self.attn_kernel_basis_size,
            activation=self.attn_activation,
            use_edge_features=self.attn_use_edge_features,
            kernel_initializer=self.attn_kernel_initializer,
            kernel_regularizer=self.attn_kernel_regularizer,
            feature_dropout=self.feature_dropout,
            edge_feature_dropout=self.edge_feature_dropout,
            batch_normalisation=self.batch_normalisation,
            name="logits",
            **kwargs)
        if self.head_aggregation == HeadAggregation.PROJECTION:
            self.projection_layer = keras_layers.Dense(
                units=self.units,
                use_bias=False,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name="projection",
                **kwargs)
        if self.batch_normalisation:
            self.batch_normalisation_layer = tf.layers.BatchNormalization()

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        if input_shape[-1].value is None:
            raise ValueError(
                'The last dimension of the inputs to `RelationalGraphConv` '
                'should be defined. Found `None`.')
        self.input_spec = InputSpec(min_ndim=2,
                                    axes={-1: input_shape[-1].value})
        self.dense_layer.build(input_shape=input_shape)
        if self.use_bias:
            bias_size = self.units
            if self.head_aggregation == HeadAggregation.CONCAT:
                bias_size = bias_size * self.heads
            self.bias = self.add_variable('bias',
                                          shape=(bias_size,),
                                          initializer=self.bias_initializer,
                                          regularizer=self.bias_regularizer,
                                          dtype=self.dtype,
                                          trainable=True)
        else:
            self.bias = None
        self.built = True

    def _attn_r_n_m_h(self):
        h, r, n = self.heads, self.relations, self._nodes

        attn_h_n_rm = self._attn_h_n_rm
        attn_h_n_r_m = tf.sparse_reshape(attn_h_n_rm, [h, n, r, n])
        attn_r_n_m_h = tf.sparse_transpose(attn_h_n_r_m, [2, 1, 3, 0])

        return attn_r_n_m_h

    def call(self, inputs, support, edge_features=None, training=False,
             attention_off=False):
        with tf.name_scope("project_features"):
            if not isinstance(inputs, tf.SparseTensor):
                inputs = tf.convert_to_tensor(inputs, dtype=self.dtype)
                self.pre_aggregation = outputs_n_rhu = self.dense_layer(inputs)
            else:
                self.pre_aggregation = outputs_n_rhu = self.dense_layer.kernel

        # support is N,RN

        # shorthands
        self._nodes = n = rgat_math_ops.get_shape(outputs_n_rhu)[0]
        r, h, u = self.relations, self.heads, self.units

        # n corresponds to `to nodes`   (of size n)
        # m corresponds to `from nodes` (of size n)

        self.logits = logits_h_n_rm = self.attention_logits(
            outputs_n_rhu, support, edge_features=edge_features,
            as_sparse_tensor=True, training=training)

        if attention_off:
            tf.logging.warning(
                "You are zeroing the attention mechanism, are you sure?")
            logits_h_n_rm = tf.SparseTensor(self.logits.indices,
                                            tf.zeros_like(logits_h_n_rm.values),
                                            self.logits.dense_shape)

        with tf.name_scope("attention_coefficients_{}".format(
                self.attention_mode)):
            if self.attention_mode == AttentionModes.ARGAT:
                attn_h_n_rm = tf.sparse_softmax(logits_h_n_rm)
            else:
                logits_h_n_r_m = tf.sparse_reshape(logits_h_n_rm,
                                                   [h, n, r, n])
                attn_h_n_r_m = tf.sparse_softmax(logits_h_n_r_m)
                attn_h_n_rm = tf.sparse_reshape(attn_h_n_r_m, [h, n, r * n])

        self._attn_h_n_rm = attn_h_n_rm

        with tf.name_scope("transform_project_features_h_rm_u"):
            outputs_m_r_h_u = tf.reshape(outputs_n_rhu, [n, r, h, u])
            outputs_h_r_m_u = tf.transpose(outputs_m_r_h_u, [2, 1, 0, 3])
            outputs_h_rm_u = self.pre_attn = tf.reshape(outputs_h_r_m_u,
                                                        [h, r * n, u])

        if self.support_dropout is not None:
            attn_values = tf.nn.dropout(attn_h_n_rm.values,
                                        keep_prob=1 - self.support_dropout,
                                        name="support_dropout")
            attn_h_n_rm = tf.SparseTensor(attn_h_n_rm.indices, attn_values,
                                          attn_h_n_rm.dense_shape)

        if self.feature_dropout is not None:
            outputs_h_rm_u = tf.nn.dropout(outputs_h_rm_u,
                                           keep_prob=1 - self.feature_dropout,
                                           name="feature_dropout")

        outputs_h_n_u = rgat_math_ops.batched_sparse_dense_matmul(
            sparse_tensor=attn_h_n_rm, dense_tensor=outputs_h_rm_u)

        with tf.name_scope("head_aggregation_{}".format(self.head_aggregation)):
            if self.head_aggregation == HeadAggregation.MEAN:
                outputs = tf.reduce_mean(outputs_h_n_u, axis=0)
            elif self.head_aggregation == HeadAggregation.SUM:
                outputs = tf.reduce_sum(outputs_h_n_u, axis=0)
            elif self.head_aggregation == HeadAggregation.CONCAT:
                outputs_n_h_u = tf.transpose(outputs_h_n_u, [1, 0, 2])
                outputs = tf.reshape(outputs_n_h_u, [n, h * u])
            elif self.head_aggregation == HeadAggregation.PROJECTION:
                outputs_n_h_u = tf.transpose(outputs_h_n_u, [1, 0, 2])
                outputs_n_hu = tf.reshape(outputs_n_h_u, [n, h * u])
                outputs = self.projection_layer(outputs_n_hu)

        if self.batch_normalisation:
            outputs = self.batch_normalisation_layer(outputs, training=training)
        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)
        if self.activation is not None:
            return self.activation(outputs)  # pylint: disable=not-callable
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if input_shape[-1].value is None:
            raise ValueError(
                'The innermost dimension of input_shape must be defined, '
                'but saw: %s'
                % input_shape)

        output_size = self.units
        if self.head_aggregation == HeadAggregation.CONCAT:
            output_size = output_size * self.heads

        return input_shape[:-1].concatenate(output_size)

    def get_config(self):
        config = {
            'units': self.units,
            'relations': self.relations,
            'heads': self.heads,
            'head_aggregation': self.head_aggregation,
            'attention_mode': self.attention_mode,
            'attention_style': self.attention_style,
            'attention_units': self.attention_units,
            'attn_use_edge_features': self.attn_use_edge_features,
            'kernel_basis_size': self.kernel_basis_size,
            'attn_kernel_basis_size': self.attn_kernel_basis_size,
            'activation': activations.serialize(self.activation),
            'attn_activation': activations.serialize(self.attn_activation),
            'use_bias': self.use_bias,
            'batch_normalisation': self.batch_normalisation,
            'kernel_initializer': initializers.serialize(
                self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'attn_kernel_initializer': initializers.serialize(
                self.attn_kernel_initializer),
            'kernel_regularizer': regularizers.serialize(
                self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'attn_kernel_regularizer': regularizers.serialize(
                self.attn_kernel_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'feature_dropout': self.feature_dropout,
            'support_dropout': self.support_dropout,
            'edge_feature_dropout': self.edge_feature_dropout
        }
        base_config = super(RelationalGraphAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# Aliases

RGAT = RelationalGraphAttention

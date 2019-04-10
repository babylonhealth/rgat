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
"""Relational graph convolution layer.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.keras import activations
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import layers as keras_layers
from tensorflow.python.keras.engine import InputSpec

from rgat import layers as rgat_layers


class RelationalGraphConv(keras_layers.Layer):
    """Layer implementing Relational Graph Convolutions with sparse supports.
    Based upon https://arxiv.org/abs/1703.06103.

    Must be called using both inputs and support:
        inputs = get_inputs()
        support = get_support()

        rgc_layer = RelationalGraphConv(...)
        outputs = rgc_layer(inputs=inputs, support=support)

    Has aliases of `RGC` and `RelationalGraphConvolution`.

    Arguments:
        units (int): The dimensionality of the output space.
        relations (int): The number of relation types the layer will handle.
        kernel_basis_size (int): The number of basis kernels to create the
            relational kernels from, i.e. W_r = sum_i c_{i,r} W'_i, where
            r = 1, 2, ..., relations, and i = 1, 2 ..., kernel_basis_size.
            If `None` (default), these is no basis decomposition.
        activation (callable): Activation function. Set it to `None` to maintain
            a linear activation.
        use_bias (bool): Whether the layer uses a bias. Defaults to `False`.
        batch_normalisation (bool): Whether the layer uses batch normalisation.
            Defaults to `False`.
        kernel_initializer (callable): Initializer function for the graph
            convolution weight matrix. If None (default), weights are
            initialized using the `glorot_uniform` initializer.
        bias_initializer (callable): Initializer function for the bias. Defaults
            to `zeros`.
        kernel_regularizer (callable): Regularizer function for the graph
            convolution weight matrix. Defaults to `None`.
        bias_regularizer (callable): Regularizer function for the bias. Defaults
            to `None`.
        activity_regularizer (callable): Regularizer function for the output.
            Defaults to `None`.
        kernel_constraint (callable): An optional projection function to be
            applied to the kernel after being updated by an Optimizer (e.g. used
            to implement norm constraints or value constraints for layer
            weights). The function must take as input the unprojected variable
            and must return the projected variable (which must have the same
            shape). Constraints are not safe to use when doing asynchronous
            distributed training.
        bias_constraint (callable): An optional projection function to be
        applied to the bias after being updated by an Optimizer.
        feature_dropout (float): The dropout rate for node feature
            representations, between 0 and 1. E.g. rate=0.1 would drop out 10%
            of node input units.
        support_dropout (float): The dropout rate for edges in the support,
            between 0 and 1. E.g. rate=0.1 would drop out 10%
            of the edges in the support.
        name (str): The name of the layer. Defaults to
            `relational_graph_conv`.

    """
    def __init__(self,
                 units,
                 relations,
                 kernel_basis_size=None,
                 activation=None,
                 use_bias=False,
                 batch_normalisation=False,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 feature_dropout=None,
                 support_dropout=None,
                 name='relational_graph_conv',
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(RelationalGraphConv, self).__init__(
            activity_regularizer=regularizers.get(activity_regularizer),
            name=name, **kwargs)

        self.units = int(units)
        self.relations = int(relations)
        self.kernel_basis_size = (int(kernel_basis_size)
                                  if kernel_basis_size is not None else None)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.batch_normalisation = batch_normalisation
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.feature_dropout = feature_dropout
        self.support_dropout = support_dropout

        self.supports_masking = True
        self.input_spec = InputSpec(min_ndim=2)

        self.dense_layer = rgat_layers.BasisDecompositionDense(
            units=self.units * self.relations,
            basis_size=self.kernel_basis_size,
            coefficients_size=self.relations,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            kernel_constraint=self.kernel_constraint,
            name=name + '_basis_decomposition_dense',
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
            self.bias = self.add_variable('bias',
                                          shape=(self.units,),
                                          initializer=self.bias_initializer,
                                          regularizer=self.bias_regularizer,
                                          constraint=self.bias_constraint,
                                          dtype=self.dtype,
                                          trainable=True)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs, support, training=False):
        if not isinstance(inputs, tf.SparseTensor):
            inputs = tf.convert_to_tensor(inputs, dtype=self.dtype)
            outputs = self.dense_layer(inputs)
        else:
            outputs = self.dense_layer.kernel                           # N,RF'
        outputs = tf.reshape(outputs, [-1, self.relations,
                                       self.units])                     # N,R,F'
        outputs = tf.transpose(outputs, perm=[1, 0, 2])                 # R,N,F'
        outputs = tf.reshape(outputs, (-1, self.units))                 # RN,F'

        if self.feature_dropout is not None:
            outputs = tf.nn.dropout(outputs,
                                    keep_prob=1 - self.feature_dropout,
                                    name="feature_dropout")
        if self.support_dropout is not None:
            support_values = tf.nn.dropout(support.values,
                                           keep_prob=1 - self.support_dropout,
                                           name="support_dropout")
            support = tf.SparseTensor(support.indices, support_values,
                                      support.dense_shape)
        outputs = tf.sparse_tensor_dense_matmul(support, outputs)       # N,F'
        if self.batch_normalisation:
            outputs = self.batch_normalisation_layer(outputs, training=training)
        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)
        if self.activation is not None:
            return self.activation(outputs)  # pylint: disable=not-callable
        return outputs

    def compute_output_shape(self, input_shape):
        return self.dense_layer.compute_output_shape(input_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'relations': self.relations,
            'rank': self.kernel_basis_size,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'batch_normalisation': self.batch_normalisation,
            'kernel_initializer': initializers.serialize(
                self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(
                self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'feature_dropout': self.feature_dropout,
            'support_dropout': self.support_dropout
        }
        base_config = super(RelationalGraphConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# Aliases

RGC = RelationalGraphConvolution = RelationalGraphConv

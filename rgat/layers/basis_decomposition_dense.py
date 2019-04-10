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
"""Basis decomposition dense layer.
"""
import tensorflow as tf

from tensorflow.python.layers import core
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base
from tensorflow.python.ops import init_ops


class BasisDecompositionDense(core.Dense):
    """Densely-connected layer class with low rank kernel given by basis
    decomposition in https://arxiv.org/abs/1703.06103.

    This layer implements the operation:
    `kernels[i] = sum_b coefficients[i, b] * basis_kernels[b]`
    `kernel = concat_1 kernels`
    `outputs = activation(inputs * kernel + bias)`
    Where `activation` is the activation function passed as the `activation`
    argument (if not `None`), `coefficients` and `basis_kernels` are weights
    matrix created by the layer and `bias` is a bias vector created by the layer
    (only if `use_bias` is `True`).

    Arguments:
      units: Integer or Long, dimensionality of the output space.
      basis_size: Integer or Long or `None`, the number of basis kernels to use.
        If `None`, there is no basis decomposition and
        `BasisDecompositionDense` behaves like a standard `Dense` layer.
      coefficients_size: Integer or Long or `None`, the number of basis
        coefficients to use. Cannot be `None` if `basis_size` is not `None`.
      activation: Activation function (callable). Set it to None to maintain a
        linear activation.
      use_bias: Boolean, whether the layer uses a bias.
      kernel_initializer: Initializer function for the weight matrix.
        If `None` (default), weights are initialized using the default
        initializer used by `tf.get_variable`.
      bias_initializer: Initializer function for the bias.
      kernel_regularizer: Regularizer function for the weight matrix.
      bias_regularizer: Regularizer function for the bias.
      activity_regularizer: Regularizer function for the output.
      kernel_constraint: An optional projection function to be applied to the
          kernel after being updated by an `Optimizer` (e.g. used to implement
          norm constraints or value constraints for layer weights). The function
          must take as input the unprojected variable and must return the
          projected variable (which must have the same shape). Constraints are
          not safe to use when doing asynchronous distributed training.
      bias_constraint: An optional projection function to be applied to the
          bias after being updated by an `Optimizer`.
      trainable: Boolean, if `True` also add variables to the graph collection
        `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
      name: String, the name of the layer. Layers with the same name will
        share weights, but to avoid mistakes we require reuse=True in such
        cases.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Properties:
      units: Python integer, dimensionality of the output space.
      basis_size: Python integer, the number of basis kernels.
      coefficients_size: Python integer, the number of basis coefficients.
      activation: Activation function (callable).
      use_bias: Boolean, whether the layer uses a bias.
      kernel_initializer: Initializer instance (or name) for the kernel matrix.
      bias_initializer: Initializer instance (or name) for the bias.
      kernel_regularizer: Regularizer instance for the kernel matrix (callable)
      bias_regularizer: Regularizer instance for the bias (callable).
      activity_regularizer: Regularizer instance for the output (callable)
      kernel_constraint: Constraint function for the kernel matrix.
      bias_constraint: Constraint function for the bias.
      kernel_left: Weight matrix (TensorFlow variable or tensor).
      kernel_right: Weight matrix (TensorFlow variable or tensor).
      kernel_diagonal: Weight matrix (TensorFlow variable or tensor).
      kernel: Composed weight matrix (TensorFlow variable or tensor).
      bias: Bias vector, if applicable (TensorFlow variable or tensor).
    """
    def __init__(self, units, basis_size, coefficients_size,
                 activation=None,
                 use_bias=True,
                 kernel_initializer=None,
                 bias_initializer=init_ops.zeros_initializer(),
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 trainable=True,
                 name=None,
                 **kwargs):
        super(BasisDecompositionDense, self).__init__(
            units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint, trainable=trainable, name=name,
            **kwargs)

        if basis_size is not None and coefficients_size is None:
            raise ValueError(
                "`coefficients_size` is {}. You must provide a "
                "`coefficients_size` for `BasisDecompositionDense`, yet your "
                "`basis_size` is {}.".format(coefficients_size, basis_size))

        self.basis_size = basis_size
        self.coefficients_size = coefficients_size

    def _build_kernel(self, input_shape):
        if self.basis_size is None:
            return self.add_variable(
                name='kernel',
                shape=[input_shape[-1].value, self.units],
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                dtype=self.dtype,
                trainable=True)

        output_shape = float(self.units) / float(self.coefficients_size)
        if not output_shape.is_integer():
            raise ValueError("units / coefficients_size is {}. "
                             "This must be an integer.".format(output_shape))
        output_shape = int(output_shape)

        self.coefficients = self.add_variable(
            name='coefficients',
            shape=[self.basis_size, self.coefficients_size],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True)
        self.kernels = self.add_variable(
            name='kernels',
            shape=[input_shape[-1].value * output_shape, self.basis_size],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True)
        kernel = tf.matmul(self.kernels, self.coefficients)
        kernel = tf.reshape(kernel, shape=[input_shape[-1].value, output_shape,
                                           self.coefficients_size])
        kernel = tf.transpose(kernel, perm=[2, 0, 1])
        return tf.reshape(kernel, [input_shape[-1].value, self.units])

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        if input_shape[-1].value is None:
            raise ValueError('The last dimension of the inputs to `Dense` '
                             'should be defined. Found `None`.')
        self.input_spec = base.InputSpec(min_ndim=2,
                                         axes={-1: input_shape[-1].value})

        self.kernel = self._build_kernel(input_shape)

        if self.use_bias:
            self.bias = self.add_variable('bias',
                                          shape=[self.units],
                                          initializer=self.bias_initializer,
                                          regularizer=self.bias_regularizer,
                                          constraint=self.bias_constraint,
                                          dtype=self.dtype,
                                          trainable=True)
        else:
            self.bias = None
        self.built = True

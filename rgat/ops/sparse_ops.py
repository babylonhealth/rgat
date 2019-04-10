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
"""Basic sparse operators for TensorFlow.
"""
import tensorflow as tf


def sparse_diagonal_matrix(sp_inputs, name="sparse_diagonal_matrix"):
    """Builds a `SparseTensor` from a list of `SparseTensor`, with the list
    appearing in order along the diagonal.

    Args:
        sp_inputs: List of `SparseTensor`.
        name: A name prefix for the returned tensors (optional).

    Returns:
        A `SparseTensor` with block diagonals corresponding to the `sp_inputs`.

    """
    if len(sp_inputs) == 1:  # Degenerate case of one tensor.
        return sp_inputs[0]

    inds = [sp_input.indices for sp_input in sp_inputs]
    vals = [sp_input.values for sp_input in sp_inputs]
    shapes = [sp_input.dense_shape for sp_input in sp_inputs]

    nmats = len(inds)

    with tf.name_scope(name):
        zeros = tf.zeros(shape=(1, 2), dtype=shapes[0].dtype)

        matrix_idxs = tf.cumsum(shapes[:-1])
        matrix_idxs = tf.concat([zeros, matrix_idxs], axis=0)
        matrix_idxs = tf.split(matrix_idxs, num_or_size_splits=nmats, axis=0)

        output_ind = [tf.add(ind, idx) for ind, idx in zip(matrix_idxs, inds)]
        output_ind = tf.concat(output_ind, axis=0)

        output_val = tf.concat(vals, axis=0)

        output_shape = tf.reduce_sum(shapes, axis=0)

        return tf.SparseTensor(output_ind, output_val, output_shape)


def sparse_squeeze_0(sp_input):
    output_ind = sp_input.indices[:, 1:]
    output_shape = sp_input.dense_shape[1:]
    return tf.SparseTensor(output_ind, sp_input.values, output_shape)


def indices_expand(indices, final_dim, name="indices_expand"):
    """Expand the indices of the final dimension of a SparseTensor.

    Takes `indices` [[0, 1],
                     [0, 2]]
    and `final_dim` 3
    to
    [[0, 1, 0],
     [0, 1, 1],
     [0, 1, 2],
     [0, 2, 0],
     [0, 2, 1],
     [0, 2, 2]]

    Args:
        indices (tf.Tensor): The indices of a `tf.SparseTensor`
        final_dim (int): The size of the final index
        name (string): Scope for this code block.

    Returns:
        (tf.Tensor) The expanded indices.

    """
    with tf.name_scope(name):
        indices_shape = tf.shape(indices)
        n_indices, indices_dim = indices_shape[0], indices_shape[1]

        extra_indices = tf.range(0, final_dim, dtype=tf.int64)
        extra_indices = tf.tile(input=extra_indices, multiples=[n_indices])
        extra_indices = tf.expand_dims(extra_indices, axis=-1)

        final_indices = tf.tile(indices, [1, final_dim])
        final_indices = tf.reshape(final_indices, [-1, indices_dim])
        final_indices = tf.concat([final_indices, extra_indices], axis=-1)

    return final_indices


def indices_expand_0(indices, zeroth_dim, name="indices_expand"):
    """Expand the first indices of of a SparseTensor.

    Takes `indices` [[0, 1],
                     [0, 2]]
    and `zeroth_dim` 3
    to
    [[0, 0, 1],
     [0, 0, 2],
     [1, 0, 1],
     [1, 0, 2],
     [2, 0, 1],
     [2, 0, 2]]

    Args:
        indices (tf.Tensor): The indices of a `tf.SparseTensor`
        zeroth_dim (int): The size of the zeroth index
        name (string): Scope for this code block.

    Returns:
        (tf.Tensor) The expanded indices.

    """
    with tf.name_scope(name):
        indices_shape = tf.shape(indices)
        n_indices = indices_shape[0]

        extra_indices = tf.range(0, zeroth_dim, dtype=tf.int64)
        extra_indices = tf.expand_dims(extra_indices, -1)
        extra_indices = tf.tile(input=extra_indices, multiples=[1, n_indices])
        extra_indices = tf.reshape(extra_indices, [-1, 1])

        final_indices = tf.tile(indices, multiples=[zeroth_dim, 1])
        final_indices = tf.concat([extra_indices, final_indices], axis=-1)

    return final_indices

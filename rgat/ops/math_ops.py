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
"""Basic arithmetic operators for TensorFlow.
"""
import tensorflow as tf


def get_shape(x):
    """
    Returns the static shape of Tensor with None values replaced with
     the corresponding dynamic values.
    x: Tensor of arbitrary shape
    """
    shape = x.shape.as_list()
    for i, value in enumerate(shape):
        if value is None:
            shape[i] = tf.shape(x)[i]  # use dynamic shape if not known
    return shape


def batched_sparse_tensor_to_sparse_block_diagonal(batched_sparse_tensor,
                                                   name=None):
    """Constructs a block-diagonal SparseTensor from batched SparseTensor
    `batched_sparse_tensor`. Each block corresponds to a batch of
    `batched_sparse_tensor` with ordering preserved.

    Args:
        batched_sparse_tensor (`SparseTensor`): Input with dense_shape
            [BATCH_SIZE, M, N]
        name: Name for this operation (optional). If not set, defaults to
            "batched_sparse_tensor_to_sparse_block_diagonal".

    Returns:
        (`SparseTensor`) Block diagonal with shape
            [BATCH_SIZE * M, BATCH_SIZE * N].
    """
    with tf.name_scope(name,
                       "batched_sparse_tensor_to_sparse_block_diagonal",
                       [batched_sparse_tensor]):
        # Dense shape of batched_sparse_tensor
        shape = tf.shape(batched_sparse_tensor,
                         out_type=tf.int64,
                         name="batched_sparse_tensor_dense_shape")

        # Calculate block-diagonal indices. The mapping is
        # (batch_num, x, y) -> (batch_num * M + x, batch_num * N + y)
        batch_nums = batched_sparse_tensor.indices[:, 0]
        offsets_x = tf.scalar_mul(shape[1], batch_nums)
        offsets_y = tf.scalar_mul(shape[2], batch_nums)
        new_indices_x = tf.add(offsets_x, batched_sparse_tensor.indices[:, 1])
        new_indices_y = tf.add(offsets_y, batched_sparse_tensor.indices[:, 2])
        indices = tf.stack([new_indices_x, new_indices_y], axis=1)

        values = batched_sparse_tensor.values
        dense_shape = (shape[0] * shape[1], shape[0] * shape[2])

        return tf.SparseTensor(
            indices=indices, values=values, dense_shape=dense_shape)


def batched_sparse_dense_matmul(sparse_tensor,
                                dense_tensor,
                                name=None):
    """
    Multiplies a batched SparseTensor and batched dense Tensor.

    Args:
        sparse_tensor: `SparseTensor` with dense_shape [BATCH_SIZE, M, N]
        dense_tensor: Dense `Tensor` with shape [BATCH_SIZE, N, D]
        name: Name for this operation (optional).
            Defaults to "batched_sparse_dense_matmul".

    Returns:
        A dense `Tensor` with shape [BATCH_SIZE, M, D]
    """
    with tf.name_scope(name,
                       "batched_sparse_dense_matmul",
                       [sparse_tensor, dense_tensor]) as scope:
        if not (isinstance(sparse_tensor, tf.SparseTensor)
                and isinstance(dense_tensor, tf.Tensor)
                and not isinstance(dense_tensor, tf.SparseTensor)):
            raise TypeError("Invalid arguments. Expected SparseTensor and "
                            "dense Tensor but got %s and %s "
                            "instead." % (sparse_tensor, dense_tensor))

        sparse_shape = tf.shape(sparse_tensor, name="sparse_shape")
        dense_shape = get_shape(dense_tensor)

        compatible_shaptes = tf.assert_equal(
            sparse_shape[2], dense_shape[1], name="assert_compatible_shapes")

        compatible_batch_sizes = tf.assert_equal(
            sparse_shape[0], dense_shape[0],
            name="assert_compatible_batch_sizes")

        with tf.control_dependencies([compatible_shaptes,
                                      compatible_batch_sizes]):

            sparse_block_diag = batched_sparse_tensor_to_sparse_block_diagonal(
                sparse_tensor, name="block_diag")

            dense_stacked = tf.reshape(
                dense_tensor,
                shape=(dense_shape[0] * dense_shape[1], dense_shape[2]),
                name="dense_stacked")

            # Multiply sparse block-diagonal with stacked dense matrix
            result_stacked = tf.sparse_tensor_dense_matmul(
                sparse_block_diag, dense_stacked, name="result_stacked")

            # Reorder 2-D result back into 3-D of shape [BATCH_SIZE, M, D]
            # TODO: use sparse_tensor.dense_shape[1] instead of -1
            result = tf.reshape(result_stacked,
                                shape=(dense_shape[0], -1, dense_shape[2]),
                                name=scope)

        return result

# Hyperparameters

## Relational Graph Attention

###  Required
+ `units (int)`: The dimensionality of the output space.
+ `relations (int)`: The number of relation types the layer will handle.

### Optional
+ `heads (int)`: The number of attention heads to use (see https://arxiv.org/abs/1710.10903). Defaults to `1`.
+ `head_aggregation` (str): The attention head aggregation method to use (see https://arxiv.org/abs/1710.10903). Can be one of `'mean'` or
            `'concat'`. Defaults to `'mean'`.
+ `attention_mode (str)`: The relational attention mode to to use (see https://openreview.net/forum?id=Bklzkh0qFm). Can be one of `'argat'`
            or `'wirgat'`. Defaults to `'argat'`.
+ `attention_style (str)`: The different types of attention to use. To use the transformer style multiplicative attention, set to `'dot'`.  To
            use the GAT style additive attention set to `'sum'`. Defaults to
            `'sum'`.
+ `attention_units (int)`: The dimensionality of the attention space. If
            using `'sum'` style attention, this must be set to `1`.
+ `attn_use_edge_features (bool)`: Whether the layer can use edge features.
            Defaults to `False`.
+ `kernel_basis_size (int)`: The number of basis kernels to create the
            relational kernels from, i.e. W_r = sum_i c_{i,r} W'_i, where
            r = 1, 2, ..., relations, and i = 1, 2 ..., kernel_basis_size.
            If `None` (default), these is no basis decomposition.
+ `attn_kernel_basis_size (int)`: The number of basis kernels to create the
            relational attention kernels from. Defaults to `None`.
+ `activation (callable)`: Activation function. Set it to `None` to maintain
            a linear activation.
+ `attn_activation (callable)`: Activation function to apply to the
            attention logits prior to feeding to softmax. Defaults to the leaky
            relu in https://arxiv.org/abs/1710.10903, however, when using
            `'dot'` style attention, this can be set to `None`.
+ `use_bias (bool)`: Whether the layer uses a bias. Defaults to `False`.
+ `batch_normalisation (bool)`: Whether the layer uses batch normalisation.
            Defaults to `False`.
+ `kernel_initializer (callable)`: Initializer function for the graph
            convolution weight matrix. If None (default), weights are
            initialized using the `glorot_uniform` initializer.
+ `bias_initializer (callable)`: Initializer function for the bias. Defaults
            to `zeros`.
+ `attn_kernel_initializer (callable)`: Initializer function for the
            attention weight matrix. If None (default), weights are
            initialized using the `glorot_uniform` initializer.
+ `kernel_regularizer (callable)`: Regularizer function for the graph
            convolution weight matrix. Defaults to `None`.
+ `bias_regularizer (callable)`: Regularizer function for the bias. Defaults
            to `None`.
+ `attn_kernel_regularizer (callable)`: Regularizer function for the graph
            attention weight matrix. Defaults to `None`.
+ `activity_regularizer (callable)`: Regularizer function for the output.
            Defaults to `None`.
+ `feature_dropout (float)`: The dropout rate for node feature
            representations, between 0 and 1. E.g. rate=0.1 would drop out 10%
            of node input units.
+ `support_dropout (float)`: The dropout rate for edges in the support,
            between 0 and 1. E.g. rate=0.1 would drop out 10%
            of the edges in the support.
+ `edge_feature_dropout (float): The dropout rate for edge feature
            representations, between 0 and 1.
+ `name (string)`: The name of the layer. Defaults to
            `rgat`.

## Relational Graph Convolution

### Required
+ `units (int)`: The dimensionality of the output space.
+ `relations (int)`: The number of relation types the layer will handle.

### Optional
+ `kernel_basis_size (int)`: The number of basis kernels to create the
            relational kernels from, i.e. W_r = sum_i c_{i,r} W'_i, where
            r = 1, 2, ..., relations, and i = 1, 2 ..., kernel_basis_size.
            If `None` (default), these is no basis decomposition.
+ `activation (callable)`: Activation function. Set it to `None` to maintain
            a linear activation.
+ `use_bias (bool)`: Whether the layer uses a bias. Defaults to `False`.
+ `batch_normalisation (bool)`: Whether the layer uses batch normalisation.
            Defaults to `False`.
+ `kernel_initializer (callable)`: Initializer function for the graph
            convolution weight matrix. If None (default), weights are
            initialized using the `glorot_uniform` initializer.
+ `bias_initializer (callable)`: Initializer function for the bias. Defaults
            to `zeros`.
+ `kernel_regularizer (callable)`: Regularizer function for the graph
            convolution weight matrix. Defaults to `None`.
+ `bias_regularizer (callable)`: Regularizer function for the bias. Defaults
            to `None`.
+ `activity_regularizer (callable)`: Regularizer function for the output.
            Defaults to `None`.
+ `kernel_constraint (callable)`: An optional projection function to be
            applied to the kernel after being updated by an Optimizer (e.g. used
            to implement norm constraints or value constraints for layer
            weights). The function must take as input the unprojected variable
            and must return the projected variable (which must have the same
            shape). Constraints are not safe to use when doing asynchronous
            distributed training.
+ `bias_constraint (callable)`: An optional projection function to be
        applied to the bias after being updated by an Optimizer.
+ `feature_dropout (float)`: The dropout rate for node feature
            representations, between 0 and 1. E.g. rate=0.1 would drop out 10%
            of node input units.
+ `support_dropout (float)`: The dropout rate for edges in the support,
            between 0 and 1. E.g. rate=0.1 would drop out 10%
            of the edges in the support.
+ `name (str)`: The name of the layer. Defaults to
            `relational_graph_conv`.

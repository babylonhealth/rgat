# Usage guide

## Calling the model
Building and calling the model is pretty simple
```python
from rgat.layers import RGAT

inputs = get_inputs()                                # Dense tensor with shape (?, Features)

support = get_support()                              # Sparse tensor with dense shape (?, ?)
support = tf.sparse_reorder(support)                 # May be neccessary, depending on construction

rgat = RGAT(units=FLAGS.units, relations=RELATIONS)  # RELATIONS is an integer indicating the number 
                                                     # of relation types in the graph

outputs = rgat(inputs=inputs, support=support)       # Dense tensor with shape (?, FLAGS.units)
```
We provide implementations for both 
[relational graph attention](https://github.com/Babylonpartners/rgat/blob/master/rgat/layers/relational_graph_attention.py#L25) and 
[relational graph convolution](https://github.com/Babylonpartners/rgat/blob/master/rgat/layers/relational_graph_convolution.py#L19) layers.
They have a plethora of hyperparameters - check out their respective docstrings for the details or alternatively [look here](hyperparameters.md)!

## Preparing supports and inputs

### Single graph
The simplest case is where we have a single graph to work on, as in the [AIFB and MUTAG example](../examples/rdf).
In this case, our features and supports tensor represents the features and support structure of a single graph.
The features are still straightfoward.
```python
inputs = get_inputs()                                      # Dense array with shape (Nodes, Features)
```
The support on the other hand should be created from constructing an `OrderdedDict` whose keys are the names of the edge type, and values are corresponding scipy sparse matrices of dense shape `(Nodes, Nodes)`
```python
support_dict = get_supports_dict()                         # Ordered dictionary   
                                                           # [('rel_1': spmatrix(...)), ...]
```
The support on the other hand should be created from constructing an `OrderdedDict` whose keys are the names of the edge type, and values are corresponding scipy sparse matrices of dense shape `(Nodes, Nodes)`
```python
support_dict = get_supports()                              # Ordered dictionary   
                                                           # [('rel_1': spmatrix(...)), ...]
```
The input into the layer is a single features tensor and a single support tensor.
To combine the `support_dict` correctly, we provide the helper function
```python
from rgat.utils import graph_utils as gu

support = gu.relational_supports_to_support(support_dict)  # Sparse tensor of dense shape
                                                           # (Nodes, Relations * Nodes) 
```                                                           
These arrays and sparse tensors can then form the basis for feeding placeholders or constructing TensorFlow datasets.
To feed a `tf.sparse.placeholder` we also provide the helper function
```python
support_triple = gu.triple_from_coo(support)               # Triple of indices, values and dense shape
```
which can then be used in
```python
support_ph = tf.sparse_placeholder(...)
feed_dict = {support_ph: support_triple, ...}
```
Don't forget to `tf.sparse_reorder` before feeding the support to the layer!
For a concrete example see the [batching example](../examples/batching).

### Multiple graphs
A more complicated scenario is where we have more than one graph to work with (for example, in a molecular property prediction task).
The features are still straightforward
```python
inputs = get_inputs()                                      # List of dense arrays, each 
                                                           # with shape (?, Features)
inputs = np.concatenate(inputs, axis=0)                    # Dense array with shape 
                                                           # (Total nodes, Features)
```
where total nodes is the number of nodes across all graphs in the input.
The support is generated from a list of `OrderdedDict`s - one for each batch element
```python
list_of_support_dicts = get_supports()                     # List of ordered dictionaries   
                                                           # [[('rel_1': spmatrix(...)), ...], ...]
```
To combine the `list_of_support_dicts` correctly, we provide the helper function
```python
support = gu.batch_of_relational_supports_to_support(      # Sparse tensor of dense shape
    support_dict)                                          # (Total nodes, Relations * Total nodes)
```                                                           
Using these `inputs` and `support` you then proceed as in the single graph case.
For a concrete example see the [batching example](../examples/batching).

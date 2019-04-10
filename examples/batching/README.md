# Batching example

## What is this?
This example shows how to use the helper functions in either a feed dict manner ([example_static.py](example_static.py)) or an eager execution manner ([example_eager.py](example_eager.py)).

In both cases, a batch of random graphs are created. A forward pass is then performed on:
+ The elements (graphs) in the batch, seperately, and
+ The elements (graphs) in the batch when they have been combined together.
The script then checks that they give the same result.

## Usage
Run from the command line (in a python environment matching TensorFlow requirements) using
```
$ python example_static.py -h

       USAGE: example_static.py [flags]
flags:

example_static.py:
  --attention_heads: The number of attention heads.
    (default: '7')
    (an integer)
  --batch_size: The batch size.
    (default: '37')
    (an integer)
  --features_dim: The input dimensionality.
    (default: '3')
    (an integer)
  --nodes_max: The largest number of nodes any graph can have. 
               This is used for random graph generation.
    (default: '9')
    (an integer)
  --nodes_min: The smallest number of nodes any graph can have. 
               This is used for random graph generation.
    (default: '2')
    (an integer)
  --relations: The number of relations.
    (default: '3')
    (an integer)
  --seed: The random seed.
    (default: '42')
    (an integer)
  --units: The number of units in the layer.
    (default: '5')
    (an integer)
```
You should see something like this
```
$ python example_static.py 
INFO:tensorflow:*************** Flags ***************
INFO:tensorflow:FLAG `seed`: 42
INFO:tensorflow:FLAG `relations`: 3
INFO:tensorflow:FLAG `nodes_min`: 2
INFO:tensorflow:FLAG `nodes_max`: 9
INFO:tensorflow:FLAG `batch_size`: 37
INFO:tensorflow:FLAG `attention_heads`: 7
INFO:tensorflow:FLAG `features_dim`: 3
INFO:tensorflow:FLAG `units`: 5
INFO:tensorflow:FLAG `h`: False
INFO:tensorflow:FLAG `help`: False
INFO:tensorflow:FLAG `helpfull`: False
INFO:tensorflow:FLAG `helpshort`: False
INFO:tensorflow:*************************************
INFO:tensorflow:Reordering indices of support - this is extremely important as sparse operations assume sparse indices have been ordered.
INFO:tensorflow:Generating support names.
INFO:tensorflow:Generating number of nodes in each element of the batch.
INFO:tensorflow:Generating fake input features for each node in each graph.
2019-01-14 17:31:50.087951: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
INFO:tensorflow:The approaches match!
```

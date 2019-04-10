# RDF example

## What is this?
This runs a two layer RGAT or RCC network on the AIFB and MUTAG node classification tasks (see https://arxiv.org/abs/1703.06103).

## Requirements
To run this example you will need `pandas`, `scipy` and `rdflib`.
The code has been sucessfully tested with the following versions
+ `pandas==0.23.4`
+ `scipy==1.2.0`
+ `rdflib==4.2.2`

although it may work with other versions. To install these, simply run
```
$ pip install -r requirements.txt
```
from this directory.
These packages are not installed by default upon installation of `rgat` as they are not neccessary for the layer definitions.

## Usage
```
$ python example.py -h

       USAGE: example.py [flags]
flags:

example.py:
  --attention_mode: The attention mode. One of `'argat'` or `'wirgat'`.
    (default: 'argat')
  --attention_style: The attention style. One of `'sum'` or `'dot'`.
    (default: 'sum')
  --attention_units: The number of units in the first attention layer. 
                     This should be `1` if we set attention style to `'sum'`
    (default: '1')
    (an integer)
  --base_dir: The base path for experiments.
    (default: '/tmp/runs')
  --dataset: The dataset. One of `'AIFB'` or `'MUTAG'`.
    (default: 'AIFB')
  --head_aggregation: The head aggregation style in the first layer. One of `'concat'` or `'mean'`.
    (default: 'concat')
  --heads: The number of attention heads in the first layer.
    (default: '4')
    (an integer)
  --learning_rate: The learning rate.
    (default: '0.01')
    (a number)
  --max_steps: The number of training steps.
    (default: '50')
    (an integer)
  --model: The model to run. One of `'rgat'` and `'rgc'`. NB, `'rgc'` is much faster.
    (default: 'rgat')
  --save_checkpoints_steps: The frequency to save checkpoints in steps.
    (default: '10')
    (an integer)
  --save_summary_steps: The frequency to save summaries in steps.
    (default: '10')
    (an integer)
  --tf_random_seed: The random seed.
    (default: '1234')
    (an integer)
  --units: The number of units in the first layer.
    (default: '16')
    (an integer)
```
You should see an output like this
```
$ python rdf/example.py
INFO:tensorflow:*************** Flags ***************
INFO:tensorflow:FLAG `base_dir`: /tmp/runs
INFO:tensorflow:FLAG `save_summary_steps`: 10
INFO:tensorflow:FLAG `save_checkpoints_steps`: 10
INFO:tensorflow:FLAG `tf_random_seed`: 1234
INFO:tensorflow:FLAG `dataset`: AIFB
INFO:tensorflow:FLAG `max_steps`: 50
INFO:tensorflow:FLAG `learning_rate`: 0.01
INFO:tensorflow:FLAG `model`: rgat
INFO:tensorflow:FLAG `units`: 16
INFO:tensorflow:FLAG `attention_units`: 1
INFO:tensorflow:FLAG `head_aggregation`: concat
INFO:tensorflow:FLAG `heads`: 4
INFO:tensorflow:FLAG `attention_style`: sum
INFO:tensorflow:FLAG `attention_mode`: argat
INFO:tensorflow:FLAG `h`: False
INFO:tensorflow:FLAG `help`: False
INFO:tensorflow:FLAG `helpfull`: False
INFO:tensorflow:FLAG `helpshort`: False
INFO:tensorflow:*************************************
INFO:tensorflow:Loading dataset AIFB.
Downloading data from https://raw.githubusercontent.com/tkipf/relational-gcn/master/rgcn/data/aifb/completeDataset.tsv
32768/28465 [==================================] - 0s 0us/step
Downloading data from https://raw.githubusercontent.com/tkipf/relational-gcn/master/rgcn/data/aifb/trainingSet.tsv
24576/22927 [================================] - 0s 0us/step
Downloading data from https://raw.githubusercontent.com/tkipf/relational-gcn/master/rgcn/data/aifb/testSet.tsv
8192/5921 [=========================================] - 0s 0us/step
Downloading data from https://www.dropbox.com/s/fkvgvkygo2gf28k/aifb_stripped.nt.gz?dl=1
573440/568040 [==============================] - 0s 0us/step
INFO:tensorflow:Did not find any preprocessed data, preprocessing...
INFO:tensorflow:Graph loaded, frequencies counted.
INFO:tensorflow:Number of nodes: 8285.
INFO:tensorflow:Number of relations: 45.
INFO:tensorflow:Building adjaceny matrix.
INFO:tensorflow:Processing relation 1/45: http://swrc.ontoware.org/ontology#abstract.
INFO:tensorflow:Processing relation 2/45: http://swrc.ontoware.org/ontology#address.
INFO:tensorflow:Processing relation 3/45: http://swrc.ontoware.org/ontology#author.
INFO:tensorflow:Processing relation 4/45: http://swrc.ontoware.org/ontology#booktitle.
```

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
"""Dataset definitions for AIFB and MUTAG semi-supervised classification tasks.
"""
import gzip
import os

import numpy as np
import pandas as pd
import rdflib as rdf
import tensorflow as tf

from collections import Counter
from scipy import sparse as sp


ALLOWED_DATASETS = {'AIFB', 'MUTAG'}

BASE_URL = (
    "https://raw.githubusercontent.com/tkipf/relational-gcn/master/rgcn/data")

BASE_URLS = {"AIFB": "aifb", "MUTAG": "mutag"}
BASE_URLS = {k: os.path.join(BASE_URL, v) for k, v in BASE_URLS.items()}

FILE_NAMES = {
    "AIFB": {
        "task": "completeDataset.tsv",
        "training_set": "trainingSet.tsv",
        "test_set": "testSet.tsv"},
    "MUTAG": {
        "task": "completeDataset.tsv",
        "training_set": "trainingSet.tsv",
        "test_set": "testSet.tsv"}}

GRAPH_URLS = {
    "AIFB": "https://www.dropbox.com/s/fkvgvkygo2gf28k/"
            "aifb_stripped.nt.gz?dl=1",
    "MUTAG": "https://www.dropbox.com/s/qy8j3p8eacvm4ir/"
             "mutag_stripped.nt.gz?dl=1"}

METADATA = {
    'AIFB': {'label_header': 'label_affiliation', 'nodes_header': 'person'},
    'MUTAG': {'label_header': 'label_mutagenic', 'nodes_header': 'bond'}}


def get_dataset_file_paths(name):
    file_names = FILE_NAMES[name]
    dataset_base_url = BASE_URLS[name]

    cache_subdir = os.path.join("datasets", name)

    def _get_file(fname):
        origin = os.path.join(dataset_base_url, fname)
        return tf.keras.utils.get_file(
            fname=fname, origin=origin, cache_subdir=cache_subdir)

    return {k: _get_file(v) for k, v in file_names.items()}


def get_graph_file_path(name):
    graph_url = GRAPH_URLS[name]
    fname = os.path.split(graph_url)[-1][:-len("?dl=1")]

    cache_subdir = os.path.join("datasets", name)

    return tf.keras.utils.get_file(
            fname=fname, origin=graph_url, cache_subdir=cache_subdir)


def _read_tsv(path):
    tf.logging.info("Reading {}.".format(path))
    return pd.read_csv(path, sep='\t', encoding='utf-8')


def get_dataset(name):
    """

    Args:
        name: Specifies the type of the RDF Dataset

    Returns:
        adjacency: Dict of Normalised Adjacency matrices (NxN)
        features: Featureless Representation of the RDF Data -
                  Identity Matrix (Nodes x Features) -> (NxN) one-hot
        labels: Sparse labels for the nodes
        train_idx: Indices of the training samples (nodes)
        test_idx: Indices of the testing samples (nodes)

    """
    tf.logging.info('Loading dataset {}.'.format(name))

    dataset_file_paths = get_dataset_file_paths(name)
    graph_file_path = get_graph_file_path(name)
    metadata = METADATA[name]

    preprocessed_path = os.path.join(
        os.path.dirname(graph_file_path), "prepro.npy")

    already_preprocessed = os.path.isfile(preprocessed_path)
    if already_preprocessed:
        tf.logging.info("Found preprocessed data. Loading from {}".format(
            preprocessed_path))
        return np.load(preprocessed_path).item()

    tf.logging.info("Did not find any preprocessed data, preprocessing...")

    preprocessed_data = _get_rdf_dataset_helper(
        dataset_file_paths=dataset_file_paths,
        graph_file_path=graph_file_path,
        metadata=metadata)

    tf.logging.info("Saving preprocessed data to {}.".format(preprocessed_path))
    np.save(preprocessed_path, preprocessed_data)
    tf.logging.info("Preprocessed data saved to {}.".format(preprocessed_path))

    return preprocessed_data


def _build_support(reader, relations, nodes, nodes_dict):
    """

    Args:
        reader:
        relations:
        nodes:
        nodes_dict:

    Returns:

    """
    tf.logging.info("Building adjaceny matrix.")
    adj_shape = (len(nodes), len(nodes))

    adjacency = dict()

    for i, rel in enumerate(sorted(relations)):
        tf.logging.info("Processing relation {}/{}: {}.".format(
            i + 1, len(relations), rel))

        edges = np.empty((reader.freq(rel), 2), dtype=np.int32)
        size = 0

        for j, (s, p, o) in enumerate(reader.triples(relation=rel)):
            if nodes_dict[s] > len(nodes) or nodes_dict[o] > len(nodes):
                tf.logging.info(s, o, nodes_dict[s], nodes_dict[o])
                raise ValueError('Relations map outside the adj matrix')

            edges[j] = np.array([nodes_dict[s], nodes_dict[o]])
            size += 1

        assert size == reader.freq(rel)

        row, col = np.transpose(edges)
        data = np.ones(len(row), dtype=np.float32)
        rel = '_'.join(os.path.basename(rel).split('#'))

        adj = sp.csr_matrix(
            (data, (row, col)), shape=adj_shape, dtype=np.float32)
        adj = normalise_matrix(adj)

        # Add the inverse relation as well
        adj_t = sp.csr_matrix(
            (data, (col, row)), shape=adj_shape, dtype=np.float32)
        adj_t = normalise_matrix(adj_t)

        adjacency[rel] = adj
        adjacency[rel + '_INV'] = adj_t

    # Add an identity adjacency matrix
    adjacency['self'] = sp.identity(adj_shape[0], dtype=np.float32).tocsr()

    return adjacency


def _get_rdf_dataset_helper(
        dataset_file_paths,
        graph_file_path,
        metadata):
    """

    Args:
        graph_file_path:
        dataset_file_paths:
        metadata:

    Returns:

    """
    reader = RDFReader(graph_file_path)

    relations = reader.relationList()
    subjects, objects = reader.subjectSet(), reader.objectSet()

    nodes = sorted(list(subjects.union(objects)))
    adj_shape = (len(nodes), len(nodes))

    tf.logging.info('Number of nodes: {}.'.format(len(nodes)))
    tf.logging.info('Number of relations: {}.'.format(len(relations)))

    nodes_dict = {node: i for i, node in enumerate(nodes)}
    assert len(nodes_dict) < np.iinfo(np.int32).max

    support = _build_support(
        reader=reader, relations=relations, nodes=nodes, nodes_dict=nodes_dict)

    nodes_u_dict = {np.unicode(to_unicode(key)): val for key, val in
                    nodes_dict.items()}

    tf.logging.info("Loading node labels.")
    labels_frames = {k: _read_tsv(v) for k, v in dataset_file_paths.items()}

    nodes_header = metadata['nodes_header']
    tf.logging.info("nodes_header: {}.".format(nodes_header))

    label_header = metadata['label_header']
    tf.logging.info("label_header: {}.".format(label_header))

    labels_set = set(labels_frames['task'][label_header].values.tolist())
    labels_dict = {lab: i for i, lab in enumerate(sorted(list(labels_set)))}
    labels = sp.lil_matrix((adj_shape[0], len(labels_set)))

    labeled_nodes_idx = list()

    def _get_indices_names(labels_df):
        nodes_values = labels_df[nodes_header].values
        labels_values = labels_df[label_header].values

        indices, names = list(), list()

        for n, l in zip(nodes_values, labels_values):
            n = np.unicode(to_unicode(n))

            if n in nodes_u_dict:
                labeled_nodes_idx.append(nodes_u_dict[n])
                idx = labels_dict[l]
                labels[labeled_nodes_idx[-1], idx] = 1
                indices.append(nodes_u_dict[n])
                names.append(n)
            else:
                tf.logging.info(u'Node not in dictionary, skipped: ',
                                n.encode('utf-8', errors='replace'))

        return indices, names

    tf.logging.info('Loading training set')
    train_idx, train_names = _get_indices_names(labels_frames['training_set'])

    tf.logging.info('Loading test set.')
    test_idx, test_names = _get_indices_names(labels_frames['test_set'])

    labels = labels.tocsr()

    # Create featureless node representations (dense matrix).
    features = sp.identity(n=adj_shape[0], dtype=np.float32)

    graph_data = {
        'support': support,
        'features': features,
        'labels': labels,
        'train_idx': train_idx,
        'test_idx': test_idx}

    return graph_data


def normalise_matrix(input_mat):
    d = np.array(input_mat.sum(1)).flatten()
    d_inv = np.zeros(d.shape, dtype=d.dtype)
    non_zero_mask = d != 0
    d_inv[non_zero_mask] = 1. / d[non_zero_mask]
    d_inv = sp.diags(d_inv)
    return d_inv.dot(input_mat).tocsr()


def to_unicode(x):
    if isinstance(x, str):
        return x
    elif isinstance(x, bytes):
        return x.decode('utf-8', errors='replace')
    else:
        raise TypeError("Cannot decode object {} of type {}.".format(
            x, type(x)))


class RDFReader:
    __graph = None
    __freq = {}

    def __init__(self, file):

        self.__graph = rdf.Graph()

        if file.endswith('nt.gz'):
            with gzip.open(file, 'rb') as f:
                self.__graph.parse(file=f, format='nt')
        else:
            self.__graph.parse(file, format=rdf.util.guess_format(file))

        self.__freq = Counter(self.__graph.predicates())

        tf.logging.info("Graph loaded, frequencies counted.")

    def triples(self, relation=None):
        for s, p, o in self.__graph.triples((None, relation, None)):
            yield s, p, o

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.__graph.destroy("store")
        self.__graph.close(True)

    def subjectSet(self):
        return set(self.__graph.subjects())

    def objectSet(self):
        return set(self.__graph.objects())

    def relationList(self):
        """
        Returns a list of relations, ordered descending by frequenecy
        :return:
        """
        res = list(set(self.__graph.predicates()))
        res.sort(key=lambda rel: - self.freq(rel))
        return res

    def __len__(self):
        return len(self.__graph)

    def freq(self, relation):
        """
        The frequency of this relation
        (how many distinct triples does it occur in?)
        :param relation:
        :return:
        """
        if relation not in self.__freq:
            return 0
        return self.__freq[relation]

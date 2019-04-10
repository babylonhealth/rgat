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
"""Some handy graph utilities for dealing with relational structures.
"""
import numpy as np

from collections import OrderedDict
from scipy import sparse


def relational_supports_to_support(relational_supports, ordering=None):
    if ordering is not None:
        return sparse.hstack([relational_supports[k] for k in ordering])

    if not isinstance(relational_supports, OrderedDict):
        ValueError("If you do not provide an ordering, then "
                   "`relational_supports` shgould be an `OrderedDict`. "
                   "It is a {}".format(type(relational_supports)))

    return sparse.hstack(list(relational_supports.values()))


def batch_of_relational_supports_to_support(batched_relational_supports,
                                            ordering=None):
    if ordering is not None:
        relational_supports_combined = OrderedDict([
            (k, sparse.block_diag([s[k] for s in batched_relational_supports]))
            for k in ordering])
    else:
        first_dict = batched_relational_supports[0]

        if not isinstance(first_dict, OrderedDict):
            ValueError("If you do not provide an ordering, then "
                       "`batched_relational_supports` shguld be list of "
                       "`OrderedDict`. It is a "
                       "list of {}".format(type(first_dict)))

        first_dict_order = list(first_dict.keys())

        relational_supports_combined = OrderedDict([
            (k, sparse.block_diag([s[k] for s in batched_relational_supports]))
            for k in first_dict_order])

    return relational_supports_to_support(relational_supports_combined,
                                          ordering=ordering)


def _indices(a):
    if len(a.data) > 0:
        return np.array(list(zip(a.row, a.col)))

    return np.empty(shape=(0, 2), dtype=np.int64)


def triple_from_coo(a):
    return _indices(a), a.data, np.array(a.shape)

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
"""Classes for validation and checking hyperparameter configurations.
"""


class HeadAggregation(object):
    """Standard names for head aggregation methods."""
    CONCAT = 'concat'
    MEAN = 'mean'
    SUM = 'sum'
    PROJECTION = 'projection'

    ALL_METHODS = {CONCAT, MEAN, SUM, PROJECTION}

    @classmethod
    def validate(cls, key):
        if key not in cls.ALL_METHODS:
            raise ValueError('Unknown head aggregation method {}. Must be '
                             'one of {}'.format(key, cls.ALL_METHODS))
        return key


class AttentionModes(object):
    """Standard names for softmax methods."""
    WIRGAT = 'wirgat'
    ARGAT = 'argat'

    ALL_METHODS = {WIRGAT, ARGAT}

    @classmethod
    def validate(cls, key):
        if key not in cls.ALL_METHODS:
            raise ValueError('Unknown attention mode {}. Must be '
                             'one of {}'.format(key, cls.ALL_METHODS))
        return key


class AttentionStyles(object):
    """Standard names for attention methods."""
    DOT = 'dot'
    SUM = 'sum'

    ALL_METHODS = {DOT, SUM}

    @classmethod
    def validate(cls, key):
        if key not in cls.ALL_METHODS:
            raise ValueError('Unknown attention mode {}. Must be '
                             'one of {}'.format(key, cls.ALL_METHODS))
        return key

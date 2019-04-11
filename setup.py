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
from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

setup(
    name='rgat',
    version='1.0.0',
    description='Relational Graph Attention Networks.',
    long_description=readme,
    author='Babylon ML team',
    author_email='dan.busbridge@babylonhealth.com',
    url='https://github.com/Babylonpartners/rgat',
    install_requires=[],
    dependency_links=[],
    packages=find_packages(exclude=('tests', 'docs', 'examples'))
)

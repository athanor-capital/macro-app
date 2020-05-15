#!/usr/bin/env python

import os
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))

required = [
    'pandas',

]

setup(
    name='macro-dashboard',
    version='0.0.1',
    packages=find_packages(),
    url='https://github.com/athanor-capital/macro-dashboards',
    author='Athanor Capital',
    author_email='shaun.viguerie@athanorcapital.com',
    install_requires=required,
)
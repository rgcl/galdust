# -*- coding: utf-8 -*-
# Copyright (C) 2019 Rodrigo González
# Licensed under the MIT licence - see LICENSE.txt
# Author: Rodrigo González-Castillo

from distutils.command.build import build

from setuptools import setup

setup(
    name='galdust',
    version='0.2',
    packages=['galdust'],

    install_requires=[
        'numpy', 'matplotlib', 'astropy', 'scipy'
    ],
    package_data={'galdust': ['galdust/data/*']},
    author='Rodrigo González-Castillo',
    author_email='rodrigo.gonzalez@uamail.cl',
    description='Models for the dust emission of galaxies using the data by the astronomer Bruce Draine',
    license='MIT',
    keywords='astrophysics, galaxy, dust emission'
)

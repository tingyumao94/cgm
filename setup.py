import os
from os.path import join as pjoin
from setuptools import setup, find_packages

setup(
    name='cgm',
    version='0.0.0',
    author='Tingyu',
    description='cgm',
    install_requires=['argparse', 'numpy', 'pyyaml', 'pandas', 'six', 'torch', 'scipy'],
    url='',
    packages=find_packages()
)
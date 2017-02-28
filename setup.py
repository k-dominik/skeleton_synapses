#!/usr/bin/env python

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


setup(name='skeleton_synapses',
      version='0.1',
      description='A tool for detecting synapses along a skeleton from CATMAID',
      author='Anna Kreshuk, Stuart Berg',
      author_email='bergs@janelia.hhmi.org',
      url='https://github.com/ilastik/catmaid_tools',
      packages=['skeleton_synapses'],
      entry_points={ 'console_scripts': ['locate_synapses = skeleton_synapses.locate_synapses:main'] }
     )

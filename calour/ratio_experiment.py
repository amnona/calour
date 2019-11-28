'''
ratio experiment (:mod:`calour.ratio_experiment`)
=======================================================

.. currentmodule:: calour.ratio_experiment

Classes
^^^^^^^
.. autosummary::
   :toctree: generated

   RatioExperiment
'''

# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from logging import getLogger

import numpy as np
import skbio

from .amplicon_experiment import AmpliconExperiment
from .util import _get_taxonomy_string, _to_list


logger = getLogger(__name__)


class RatioExperiment(AmpliconExperiment):
    '''This class stores log-ratio values for each amplicon feature in different conditions

    This is a child class of :class:`.Experiment`

    Parameters
    ----------
    data : numpy.ndarray or scipy.sparse.csr_matrix
        The ratio table for OTUs, metabolites, genes, etc. Samples
        are in row and features in column
    sample_metadata : pandas.DataFrame
        The metadata on the samples
    feature_metadata : pandas.DataFrame
        The metadata on the features
    description : str
        name of experiment
    sparse : bool
        store the data array in :class:`scipy.sparse.csr_matrix`
        or :class:`numpy.ndarray`

    Attributes
    ----------
    data : numpy.ndarray or scipy.sparse.csr_matrix
        The ratio table for OTUs, metabolites, genes, etc. Samples
        are in row and features in column
    sample_metadata : pandas.DataFrame
        The metadata on the samples
    feature_metadata : pandas.DataFrame
        The metadata on the features
    exp_metadata : dict
        metadata about the experiment (data md5, filenames, etc.)
    shape : tuple of (int, int)
        the dimension of data
    sparse : bool
        store the data as sparse matrix (scipy.sparse.csr_matrix) or dense numpy array.
    description : str
        name of the experiment

    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def plot(self, *args, **kwargs):
        '''Override the default plot() function to remove the log normalization, and change default colormap and clim
        '''
        if 'norm' not in kwargs:
            kwargs['norm'] = None
        if 'cmap' not in kwargs:
            kwargs['cmap'] = 'coolwarm'
        if 'clim' not in kwargs:
            kwargs['clim'] = [-1, 1]
        super().plot(self, *args, **kwargs)

    def sort_abundance(self, *args, **kwargs):
        '''Override the default sort_abundance() function to remove log normalization
        '''
        if 'key' not in kwargs:
            kwargs['key'] = np.mean
        super().sort_abundance(self, *args, **kwargs)

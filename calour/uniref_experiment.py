'''
UniRef experiment (:mod:`calour.uniref_experiment`)
=======================================================

.. currentmodule:: calour.uniref_experiment

Classes
^^^^^^^
.. autosummary::
   :toctree: generated

   UniRefExperiment
'''

# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from logging import getLogger
from copy import deepcopy
from collections import defaultdict

import numpy as np
import matplotlib as mpl
import pandas as pd

from .experiment import Experiment
from .io import read
from .util import _get_taxonomy_string, _to_list
from .database import _get_database_class
from .experiment import Experiment


logger = getLogger(__name__)


class UniRefExperiment(Experiment):
    '''This class stores UniRef experiment
    Interactive heatmap uniref information is obtained through the uniref_calour module

    This is a child class of :class:`.Experiment`.

    Parameters
    ----------
    data : numpy.ndarray or scipy.sparse.csr_matrix
        The expression table for genes. Samples
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
    databases: iterable of str, optional
        database interface names to show by default in heatmap() function
        by default use 'dbbact'

    Attributes
    ----------
    data : numpy.ndarray or scipy.sparse.csr_matrix
        The expression table for genes. Samples
        are in row and features in column
    sample_metadata : pandas.DataFrame
        The metadata on the samples
    feature_metadata : pandas.DataFrame
        The metadata on the features
    shape : tuple of (int, int)
        the dimension of data
    sparse : bool
        store the data as sparse matrix (scipy.sparse.csr_matrix) or dense numpy array.
    info : dict
        information about the experiment (data md5, filenames, etc.)
    description : str
        name of the experiment
    databases : dict
        keys are the database names (i.e. 'dbbact' / 'gnps')
        values are the database specific data for the experiment (i.e. annotations for dbbact)

    See Also
    --------
    Experiment
    '''
    def __init__(self, *args, databases=(), **kwargs):
        super().__init__(*args, databases=('uniref',), **kwargs)

    def heatmap(self, *args, **kwargs):
        '''Plot a heatmap for the uniref experiment.

        This method accepts exactly the same parameters as input with
        its parent class method and does exactly the sample plotting.

        The only difference is that by default, its color scale is **in
        log** as its `norm` parameter is set to
        `matplotlib.colors.LogNorm()`. It makes more sense to show the
        gene expression abundances in color of log scale since they cover a wide range of magnitudes.
        You can always set it to other scale as
        explained in :meth:`.Experiment.heatmap`.

        Parameters
        ----------

        Keyword Arguments
        -----------------
        %(experiment.heatmap.parameters)s

        See Also
        --------
        Experiment.heatmap
        '''
        # set this default value inside the function instead of on the
        # function API (like the __init__) because we don't wanna to
        # define mpl.colors.LogNorm() on the API; otherwise, vmin and
        # vmax are set the same once for all UniRefExperiment
        # objects (which we don't want) because python initializes
        # the function arguments when it reads in its definition.

        # by default use the log normalization
        if 'norm' not in kwargs:
            kwargs['norm'] = mpl.colors.LogNorm()
        super().heatmap(*args, **kwargs)

    @staticmethod
    def read(*kargs, **kwargs):
        '''Load a UniRef experiment. calls calour.io.read() providing the correct class parameter (cls=UniRefExperiment).
        by default, the UniRefExperiment table is expected to be tab separated (can modify by the setting data_file_sep parameter),
        and samples are in columns (can modify by setting sample_in_row parameter).
        By default, the data is not normalized. To normalize the per-sample reads to sum X, set normalize=X.
        For more details, see 

        Parameters
        ----------

        Keyword Arguments
        -----------------
        %(io.read.parameters)s

        Returns
        -------
        ca.UniRefExperiment

        See Also
        --------
        calour.io.read
        '''
        if 'data_file_sep' not in kwargs:
            kwargs['data_file_sep'] = '\t'
        if 'sparse' not in kwargs:
            kwargs['sparse'] = False
        if 'sample_in_row' not in kwargs:
            kwargs['sample_in_row'] = False
        if 'normalize' not in kwargs:
            kwargs['normalize'] = None

        dat = read(*kargs, **kwargs, cls=UniRefExperiment)
        return dat
    

    def fetch_uniref_info(self):
        '''Prefetch the uniref information for the features in the experiment and add it to the feature metadata.
        This will add the per-unirefid information to the local database, so it will be available for other datasets/rerunning the same dataset without the need to fetch it again from the remote servers.
        '''
        from tqdm import tqdm

        db = _get_database_class('uniref')
        for x in tqdm(self.feature_metadata.index.values):
            cid = x.split('.')[0]
            res=db._get_uniref_info(cid)

    
    def aggregate_uniref(self, aggregate=['name: ', 'go term: ']):
        '''Aggregate the features in the experiment by info from uniref annotations
        It will add the number of observations of each feature to each name/go term it appears in, and create a new experiment with the aggregated data.
    
        Parameters
        ----------
        aggregate: list of str
            the uniref annotation types to aggregate by. 
            Options include 'name: ' (common name), 'go term: ' (gene ontology term), 'organism: '
        
        Returns
        -------
        calour.Experiment
             a new experiment (Samples * aggregated_features) with the aggregated data. The feature metadata have the columns:
             name: the name of the annotation (e.g. go term, etc.)
             uniref_ids: the list of uniref ids that were aggregated
        '''
        db=_get_database_class('uniref')
        all_names = defaultdict(list)
        for cid,x in self.feature_metadata.iterrows():
            annotations=db.get_seq_annotation_strings(cid)
            for canno in annotations:
                cname = None
                for agg in aggregate:
                    if canno[1].startswith(agg):
                        cname = canno[1][len(agg):]
                        all_names[cname].append(cid)
                        break
                if cname is None:
                    continue
                all_names[cname].append(cid)

        # print the top common names
        logger.info('found %d unique names' % len(all_names))
        for cname, cids in sorted(all_names.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
            logger.debug('%s: %d', cname, len(cids))

        # create new names experiment
        names_dat = np.zeros((self.shape[0], len(all_names)), dtype=float)
        data = self.get_data(sparse=False)
        for i, (cname, cids) in enumerate(all_names.items()):
            for cid in cids:
                cid_pos = [self.feature_metadata.index.get_loc(cid)][0]
                names_dat[:, i] += data[:, cid_pos]
        fmd = pd.DataFrame([all_names.keys(), all_names.values()], columns=['name', 'uniref_ids'])
        names_dat_exp = Experiment(names_dat, self.sample_metadata.copy(), fmd, description=f'created from {self.description} by function_exp_from_uniref', sparse=False)
        return names_dat_exp

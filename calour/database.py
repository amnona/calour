from logging import getLogger
from abc import ABC

logger = getLogger(__name__)


class Database(ABC):
    def __init__(self, database_name='generic', methods=['get', 'annotate', 'feature_terms']):
        '''Initialize the database interface

        Parameters
        ----------
        database_name : str (optional)
            name of the database
        methods : list of str (optional)
            'get' if database interface supports get_seq_annotation_strings()
            'annotate' if database interface supports add_annotation()
            'enrichment' if database interface supports get_feature_terms()
        '''
        self._database_name = database_name
        self._methods = set(methods)

    def get_name(self):
        '''Get the name of the database.
        Used for displaying when no annotations are found

        Returns
        -------
        dbname : str
            nice name of the database
        '''
        return self._database_name

    @property
    def annotatable(self):
        '''True if the database supports adding annotations via the add_annotation() function
        '''
        return 'annotate' in self._methods

    @property
    def can_get_feature_terms(self):
        '''True if the database supports getting a dict of terms per feature via the get_feature_terms() function
        '''
        return 'feature_terms' in self._methods

    def get_seq_annotation_strings(self, sequence):
        '''Get nice string summaries of annotations for a given sequence

        Parameters
        ----------
        sequence : str
            the DNA sequence to query the annotation strings about

        Returns
        -------
        shortdesc : list of (dict,str) (annotationdetails,annotationsummary)
            a list of:
                annotationdetails : dict
                    'annotationid' : int, the annotation id in the database
                    'annotationtype : str
                    ...
                annotationsummary : str
                    a short summary of the annotation
        '''
        logger.debug('Generic function for get_annotation_strings')
        return []

    def show_annotation_info(self, annotation):
        '''Show details about the annotation

        Parameters
        ----------
        annotation : dict
            See dbBact REST API /annotations/get_annotation for keys / values
        '''
        # open in a new tab, if possible
        logger.debug('Generic function for show annotation info')
        return

    def add_annotation(self, features, exp):
        '''Add an entry to the database about a set of features

        Parameters
        ----------
        features : list of str
            the features to add to the database
        exp : calour.Experiment
            the experiment where the features are coming from

        Returns
        -------
        err : str
            empty if ok, otherwise the error encountered
        '''
        logger.debug('Generic function for add_annotations')
        raise NotImplemented

    def delete_annotation(self, annotation_details):
        '''Delete an annotation from the database (if allowed)

        Parameters
        ----------
        annotation_details : dict
            The details about the annotation to delete (annotationdetails from get_seq_annotation_strings() )
            Should contain a unique identifier for the annotation (created/used by the database)

        Returns
        -------
        str
            empty if ok, otherwise the error encountered
        '''
        logger.debug('Generic function for delete_annotation')
        return 'Not implemented'

    def remove_feature_from_annotation(self, features, annotation_details):
        '''remove a feature from the annotation in the database (if allowed)

        Parameters
        ----------
        features : list of str
            The feature ids to remove
        annotation_details : dict
            The details about the annotation to delete (annotationdetails from get_seq_annotation_strings() )
            Should contain a unique identifier for the annotation (created/used by the database)

        Returns
        -------
        str
            empty if ok, otherwise the error encountered
        '''
        logger.debug('Generic function for remove_features_from_annotation')
        return 'Not implemented'

    def get_feature_terms(self, features, exp=None):
        '''Get list of terms per feature

        Parameters
        ----------
        features : list of str
            the features to get the terms for
        exp : calour.Experiment (optional)
            not None to store results inthe exp (to save time for multiple queries)

        Returns
        -------
        feature_terms : dict of list of str/int
            key is the feature, list contains all terms associated with the feature
        '''
        logger.debug('Generic function for get_feature_terms')
        return {}

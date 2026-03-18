# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from unittest import TestCase, main
from unittest.mock import patch

import numpy as np
import pandas as pd
import matplotlib as mpl

from calour.experiment import Experiment
from calour.mrna_experiment import mRNAExperiment


class TestMRNAExperiment(TestCase):
    def setUp(self):
        self.data = np.array([[1.0, 2.0], [3.0, 4.0]])
        self.sample_metadata = pd.DataFrame(index=['s1', 's2'])
        self.feature_metadata = pd.DataFrame(index=['g1', 'g2'])

    def test_init_sets_mrna_database(self):
        exp = mRNAExperiment(self.data, self.sample_metadata, self.feature_metadata, sparse=False)

        self.assertIn('mrna', exp.databases)

    def test_heatmap_sets_default_lognorm(self):
        exp = mRNAExperiment(self.data, self.sample_metadata, self.feature_metadata, sparse=False)

        with patch.object(Experiment, 'heatmap', autospec=True, return_value='mock-ax') as mock_heatmap:
            exp.heatmap()

        self.assertEqual(mock_heatmap.call_count, 1)
        call_kwargs = mock_heatmap.call_args.kwargs
        self.assertIn('norm', call_kwargs)
        self.assertIsInstance(call_kwargs['norm'], mpl.colors.LogNorm)

    def test_heatmap_preserves_explicit_norm(self):
        exp = mRNAExperiment(self.data, self.sample_metadata, self.feature_metadata, sparse=False)
        custom_norm = mpl.colors.Normalize(vmin=0, vmax=10)

        with patch.object(Experiment, 'heatmap', autospec=True, return_value='mock-ax') as mock_heatmap:
            exp.heatmap(norm=custom_norm)

        call_kwargs = mock_heatmap.call_args.kwargs
        self.assertIs(call_kwargs['norm'], custom_norm)

    def test_read_sets_defaults_and_class(self):
        with patch('calour.mrna_experiment.read', autospec=True, return_value='mock-exp') as mock_read:
            result = mRNAExperiment.read(data_file='dummy.tsv', sample_metadata_file='dummy.map')

        self.assertEqual(result, 'mock-exp')
        self.assertEqual(mock_read.call_count, 1)
        call_kwargs = mock_read.call_args.kwargs
        self.assertEqual(call_kwargs['data_file_sep'], '\t')
        self.assertEqual(call_kwargs['sparse'], False)
        self.assertEqual(call_kwargs['sample_in_row'], False)
        self.assertIsNone(call_kwargs['normalize'])
        self.assertIs(call_kwargs['cls'], mRNAExperiment)
        self.assertEqual(call_kwargs['data_file'], 'dummy.tsv')
        self.assertEqual(call_kwargs['sample_metadata_file'], 'dummy.map')

    def test_read_keeps_user_overrides(self):
        with patch('calour.mrna_experiment.read', autospec=True, return_value='mock-exp') as mock_read:
            mRNAExperiment.read(
                data_file='dummy.tsv',
                sample_metadata_file='dummy.map',
                data_file_sep=',',
                sparse=True,
                sample_in_row=True,
                normalize=1000,
            )

        call_kwargs = mock_read.call_args.kwargs
        self.assertEqual(call_kwargs['data_file_sep'], ',')
        self.assertEqual(call_kwargs['sparse'], True)
        self.assertEqual(call_kwargs['sample_in_row'], True)
        self.assertEqual(call_kwargs['normalize'], 1000)
        self.assertIs(call_kwargs['cls'], mRNAExperiment)


if __name__ == '__main__':
    main()
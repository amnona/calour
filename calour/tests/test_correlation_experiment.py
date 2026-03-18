# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from unittest import main

import numpy as np
import pandas as pd

import calour as ca
from calour._testing import Tests
from calour.correlation_experiment import CorrelationExperiment


class TestCorrelationExperiment(Tests):
    def setUp(self):
        super().setUp()
        self.test1 = ca.read(self.test1_biom, self.test1_samp, normalize=None)

    def test_calculate_corr_matrix_basic(self):
        df1 = pd.DataFrame({
            'a': [1, 2, 3, 4],
            'b': [4, 3, 2, 1],
        })
        df2 = pd.DataFrame({
            'c': [10, 20, 30, 40],
            'd': [40, 30, 20, 10],
        })

        corrs, pvals = CorrelationExperiment._calculate_corr_matrix(df1, df2, add_noise=0)

        np.testing.assert_allclose(corrs, np.array([[1.0, -1.0], [-1.0, 1.0]]))
        self.assertEqual(corrs.shape, (2, 2))
        self.assertEqual(pvals.shape, (2, 2))

    def test_calculate_corr_matrix_constant_column_sets_default_values(self):
        df1 = pd.DataFrame({'a': [1, 1, 1, 1]})
        df2 = pd.DataFrame({'c': [1, 2, 3, 4]})

        corrs, pvals = CorrelationExperiment._calculate_corr_matrix(df1, df2, add_noise=0)

        self.assertEqual(corrs[0, 0], 0)
        self.assertEqual(pvals[0, 0], 1)

    def test_from_dataframes_single_dataframe(self):
        df = pd.DataFrame({
            'x': [1, 2, 3, 4],
            'y': [4, 3, 2, 1],
            'z': [1, 1, 2, 2],
        })

        exp = CorrelationExperiment.from_dataframes(df, add_noise=0)

        self.assertIsInstance(exp, CorrelationExperiment)
        self.assertEqual(exp.shape, (3, 3))
        self.assertEqual(exp.qvals.shape, (3, 3))
        self.assertEqual(set(exp.sample_metadata.index), set(df.columns))
        self.assertEqual(set(exp.feature_metadata.index), set(df.columns))

    def test_from_data_invalid_axis_raises(self):
        with self.assertRaises(ValueError):
            CorrelationExperiment.from_data(self.test1, axis='invalid')

    def test_filter_qvals_strict_and_non_strict(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        qvals = np.array([[0.01, 0.05], [0.2, 0.8]])
        smd = pd.DataFrame(index=['s1', 's2'])
        fmd = pd.DataFrame(index=['f1', 'f2'])
        exp = CorrelationExperiment(data=data, sample_metadata=smd, feature_metadata=fmd, qvals=qvals, sparse=False)

        filtered_strict = exp.filter_qvals(threshold=0.05, strict=True, replace_value=-999, inplace=False)
        np.testing.assert_array_equal(filtered_strict.data, np.array([[1.0, 2.0], [-999.0, -999.0]]))
        np.testing.assert_array_equal(exp.data, data)

        filtered_nonstrict = exp.filter_qvals(threshold=0.05, strict=False, replace_value=-777, inplace=False)
        np.testing.assert_array_equal(filtered_nonstrict.data, np.array([[1.0, -777.0], [-777.0, -777.0]]))

    def test_filter_qvals_without_qvals_raises(self):
        data = np.array([[1.0]])
        smd = pd.DataFrame(index=['s1'])
        fmd = pd.DataFrame(index=['f1'])
        exp = CorrelationExperiment(data=data, sample_metadata=smd, feature_metadata=fmd, sparse=False)

        with self.assertRaises(ValueError):
            exp.filter_qvals()


if __name__ == '__main__':
    main()
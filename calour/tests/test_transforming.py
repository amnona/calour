# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

import sys
from unittest import main, skipIf

import numpy as np
import pandas as pd
from numpy.testing import assert_array_almost_equal, assert_array_equal

import calour as ca
from calour._testing import Tests


class TestTransforming(Tests):
    def setUp(self):
        super().setUp()
        self.test2 = ca.read(self.test2_biom, self.test2_samp, self.test2_feat, normalize=None)

    def test_standardize(self):
        obs = self.test2.standardize()
        self.assertIsNot(obs, self.test2)
        assert_array_almost_equal(obs.data.sum(axis=1), [0] * 9)
        assert_array_almost_equal(obs.data.var(axis=1), [1] * 9)
        obs = self.test2.standardize(inplace=True)
        self.assertIs(obs, self.test2)

    def test_binarize(self):
        obs = self.test2.binarize()
        self.assertIsNot(obs, self.test2)
        obs = self.test2.binarize(inplace=True)
        self.assertIs(obs, self.test2)

    def test_log_n(self):
        obs = self.test2.log_n()
        self.test2.data = np.log2(
            [[10., 20., 1., 20., 5., 100., 844., 100.],
             [10., 20., 2., 19., 1., 100., 849., 200.],
             [10., 20., 3., 18., 5., 100., 844., 300.],
             [10., 20., 4., 17., 1., 100., 849., 400.],
             [10., 20., 5., 16., 4., 100., 845., 500.],
             [10., 20., 6., 15., 1., 100., 849., 600.],
             [10., 20., 7., 14., 3., 100., 846., 700.],
             [10., 20., 8., 13., 1., 100., 849., 800.],
             [10., 20., 9., 12., 7., 100., 842., 900.]])
        self.assert_experiment_equal(obs, self.test2)
        self.assertIsNot(obs, self.test2)

        obs = self.test2.log_n(inplace=True)
        self.assertIs(obs, self.test2)

    def test_center_log_ration(self):
        from skbio.stats.composition import clr, centralize

        dat = np.array(
            [[10, 20, 1, 20, 5, 100, 844, 100],
             [10, 20, 2, 19, 0, 100, 849, 200],
             [10, 20, 3, 18, 5, 100, 844, 300],
             [10, 20, 4, 17, 0, 100, 849, 400],
             [10, 20, 5, 16, 4, 100, 845, 500],
             [10, 20, 6, 15, 0, 100, 849, 600],
             [10, 20, 7, 14, 3, 100, 846, 700],
             [10, 20, 8, 13, 0, 100, 849, 800],
             [10, 20, 9, 12, 7, 100, 842, 900]]) + 1
        obs = self.test2.center_log_ratio()
        exp = clr(dat)
        assert_array_almost_equal(exp, obs.data)
        obs = self.test2.center_log_ratio(centralize=True)
        exp = clr(centralize(dat))
        assert_array_almost_equal(exp, obs.data)

    def test_normalize(self):
        total = 1000
        obs = self.test2.normalize(total)
        assert_array_almost_equal(obs.data.sum(axis=1).A1,
                                  [total] * 9)
        self.assertIsNot(obs, self.test2)

        obs = self.test2.normalize(total, inplace=True)
        self.assertIs(obs, self.test2)

    def test_normalize_non_numeric(self):
        with self.assertRaises(ValueError):
            self.test2.normalize(False)

    def test_rescale(self):
        total = 1000
        obs = self.test2.rescale(total)
        self.assertAlmostEqual(np.mean(obs.data.sum(axis=1)), 1000)
        self.assertIsNot(obs, self.test2)
        self.assertNotAlmostEqual(obs.data.sum(axis=1).A1[0], 1000)

    def test_rescale_non_numeric(self):
        with self.assertRaises(ValueError):
            self.test2.normalize(False)
        with self.assertRaises(ValueError):
            self.test2.normalize(0)

    def test_normalize_by_subset_features(self):
        # test the filtering in standard mode (remove a few features, normalize to 10k)
        exp = ca.read(self.test1_biom, self.test1_samp, normalize=None)
        bad_features = [6, 7]
        features = [exp.feature_metadata.index[cbad] for cbad in bad_features]
        newexp = exp.normalize_by_subset_features(features, 10000, negate=True, inplace=False)
        # see the mean of the features we want (without 6,7) is 10k
        good_features = list(set(range(exp.data.shape[1])).difference(set(bad_features)))
        assert_array_almost_equal(newexp.data[:, good_features].sum(axis=1), np.ones([exp.data.shape[0]]) * 10000)
        self.assertTrue(np.all(newexp.data[:, bad_features] > exp.data[:, bad_features]))

    def test_permute_data(self):
        exp = ca.Experiment(data=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                            sample_metadata=pd.DataFrame(index=['s1', 's2', 's3']),
                            feature_metadata=pd.DataFrame(index=['f1', 'f2', 'f3']),
                            sparse=False)

        obs = exp.permute_data(normalize=False, random_seed=123)
        self.assertEqual(obs.shape, exp.shape)
        for cfeature in range(exp.shape[1]):
            self.assertCountEqual(list(obs.data[:, cfeature]), list(exp.data[:, cfeature]))

        obs_norm = exp.permute_data(normalize=True, random_seed=123)
        self.assertAlmostEqual(np.std(obs_norm.data.sum(axis=1)), 0)

    def test_normalize_compositional(self):
        exp = ca.read(self.test1_biom, self.test1_samp, normalize=None)
        obs = exp.normalize_compositional(frac=0.1, total=5000)
        comp_features = exp.normalize().filter_mean_abundance(0.1)
        use_mask = ~obs.feature_metadata.index.isin(comp_features.feature_metadata.index.values)
        assert_array_almost_equal(obs.data[:, use_mask].sum(axis=1), np.ones([obs.shape[0]]) * 5000)

    def test_subsample_count_replace(self):
        exp = ca.Experiment(data=np.array([[1, 2, 3], [4, 5, 6]]),
                            sample_metadata=pd.DataFrame(index=['s1', 's2']),
                            feature_metadata=pd.DataFrame(index=['f1', 'f2', 'f3']),
                            sparse=False)
        obs = exp.subsample_count(10, replace=True, random_seed=1)
        assert_array_equal(obs.data.sum(axis=1), np.array([10, 10]))

    def test_subsample_count_non_integer_raises(self):
        exp = ca.Experiment(data=np.array([[1.1, 2.0, 3.0], [4.0, 5.0, 6.0]]),
                            sample_metadata=pd.DataFrame(index=['s1', 's2']),
                            feature_metadata=pd.DataFrame(index=['f1', 'f2', 'f3']),
                            sparse=False)
        with self.assertRaises(ValueError):
            exp.subsample_count(5)

    @skipIf(sys.platform.startswith("win"), "skip this test for Windows")
    def test_subsample_count(self):
        exp = ca.Experiment(data=np.array([[1, 2, 3], [4, 5, 6]]),
                            sample_metadata=pd.DataFrame([['a', 'b', 'c'], ['d', 'e', 'f']]),
                            sparse=False)
        n = 6
        obs = exp.subsample_count(n, random_seed=9)
        assert_array_equal(obs.data.sum(axis=1), np.array([n, n]))
        self.assertTrue(np.all(obs.data <= n))

        n = 7
        obs = exp.subsample_count(n)
        # the 1st row dropped
        assert_array_equal(obs.data.sum(axis=1), np.array([n]))
        self.assertIsNot(obs, exp)

        obs = exp.subsample_count(n, inplace=True)
        assert_array_equal(obs.data.sum(axis=1), np.array([n]))
        self.assertTrue(np.all(obs.data <= n))
        self.assertIs(obs, exp)

        n = 10000
        exp.normalized = False
        obs = exp.subsample_count(n)
        assert_array_equal(obs.data.sum(axis=1), np.array([]))


if __name__ == '__main__':
    main()

# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

import unittest

import numpy as np
import pandas.testing as pdt

import calour as ca
from calour.analysis import diff_abundance, diff_abundance_paired, correlation_paired, _prepare_paired
from calour._testing import Tests


class TestAnalysis(Tests):
    def setUp(self):
        super().setUp()
        # load the simple experiment as sparse
        self.test1 = ca.read(self.test1_biom, self.test1_samp, normalize=None)
        # load the paired testing experiment
        self.test_paired = ca.read(self.test_paired_biom, self.test_paired_samp, normalize=None)
        # load the complex experiment as sparse with normalizing and removing low read samples
        self.complex = ca.read_amplicon(self.timeseries_biom, self.timeseries_samp,
                                        min_reads=1000, normalize=10000)

    def test_diff_abundance(self):
        # test using defulat values
        dd = diff_abundance(self.test1, 'group', val1='1', val2='2', random_seed=2017)
        expected_ids = [0, 1, 2, 3, 4, 7, 10]
        self.assertEqual(len(dd.feature_metadata), 7)
        for cid in expected_ids:
            self.assertIn(self.test1.feature_metadata.index[cid], dd.feature_metadata.index)
        # test using no group 2 using only group 1
        dd = diff_abundance(self.test1, 'group', val1='1')
        expected_ids = [0, 1, 2, 3, 4, 10]
        # we get 1 less since now we also include badsample sample (not in the mapping file, so gets na)
        self.assertEqual(len(dd.feature_metadata), 6)
        for cid in expected_ids:
            self.assertIn(self.test1.feature_metadata.index[cid], dd.feature_metadata.index)
        # test using no val 2 using only group 2
        dd = diff_abundance(self.test1, 'group', val1='2')
        expected_ids = [0, 1, 2, 3, 4, 7, 10]
        self.assertEqual(len(dd.feature_metadata), 7)
        for cid in expected_ids:
            self.assertIn(self.test1.feature_metadata.index[cid], dd.feature_metadata.index)
    
    def test_diff_abundance_fast_vs_slow(self):
        dd_fast = diff_abundance(self.test1, 'group', val1='1', val2='2', random_seed=2017, method='meandiff')
        dd_slow = diff_abundance(self.test1, 'group', val1='1', val2='2', random_seed=2017, method=ca.dsfdr.meandiff)
        # assert whether the XX.feature_metadata DataFrames as equal
        pdt.assert_frame_equal(dd_fast.feature_metadata, dd_slow.feature_metadata,check_like=True,check_exact=False, atol=1e-2)

    def test_diff_abundance_alpha0(self):
        # Test when we should get 0 features (setting FDR level to 0)
        dd = diff_abundance(self.test1, 'group', val1='1', val2='2', alpha=0)
        self.assertEqual(dd.shape, (self.test1.shape[0], 0))

    def test_correlation_default(self):
        # test using spearman correlation
        dd = self.test1.correlation('id', random_seed=2017)
        expected_ids = [0, 1, 2, 3, 4, 7, 10]
        self.assertEqual(len(dd.feature_metadata), 7)
        for cid in expected_ids:
            self.assertIn(self.test1.feature_metadata.index[cid], dd.feature_metadata.index)

    def test_correlation_nonzero_spearman(self):
        # test using non zero spearman correlation
        dd = self.test1.correlation('id', method='spearman', nonzero=True, random_seed=2017)
        expected_ids = [1, 2, 4, 5, 7, 10]
        self.assertEqual(len(dd.feature_metadata), 6)
        for cid in expected_ids:
            self.assertIn(self.test1.feature_metadata.index[cid], dd.feature_metadata.index)

    def test_correlation_pearson(self):
        # test using pearson correlation
        dd = self.test1.correlation('id', method='pearson', random_seed=2017)
        expected_ids = [0, 1, 2, 3, 4, 7, 10]
        self.assertEqual(len(dd.feature_metadata), 7)
        for cid in expected_ids:
            self.assertIn(self.test1.feature_metadata.index[cid], dd.feature_metadata.index)

    def test_correlation_nonzero_pearson(self):
        # test using non zero pearson correlation
        dd = self.test1.correlation('id', method='pearson', nonzero=True, random_seed=2017)
        expected_ids = [1, 2, 4, 5, 7, 10]
        self.assertEqual(len(dd.feature_metadata), 6)
        for cid in expected_ids:
            self.assertIn(self.test1.feature_metadata.index[cid], dd.feature_metadata.index)

    def test_correlation_complex(self):
        # test on real complex dataset (timeseries)
        # after rank transforming the reads, should get
        dd = self.complex.correlation('MF_SAMPLE_NUMBER', method='pearson', transform='rankdata', random_seed=2017)
        self.assertEqual(len(dd.feature_metadata), 46)
        goodseq = 'TACGGAGGATGCGAGCGTTATTCGGAATCATTGGGTTTAAAGGGTCTGTAGGCGGGCTATTAAGTCAGGGGTGAAAGGTTTCAGCTTAACTGAGAAATTGCCTTTGATACTGGTAGTCTTGAATATCTGTGAAGTTCTTGGAATGTGTAG'
        self.assertIn(goodseq, dd.feature_metadata.index)
        goodseq = 'TACGTAGGTGGCAAGCGTTGTCCGGAATTATTGGGCGTAAAGCGCGCGCAGGCGGATCAGTCAGTCTGTCTTAAAAGTTCGGGGCTTAACCCCGTGATGGGATGGAAACTGCTGATCTAGAGTATCGGAGAGGAAAGTGGAATTCCTAGT'
        self.assertIn(goodseq, dd.feature_metadata.index)
        # with no transform
        dd = self.complex.correlation('MF_SAMPLE_NUMBER', method='pearson', random_seed=2017)
        # print(len(dd.feature_metadata))
        # print(dd.feature_metadata)
        # self.assertTrue(np.abs(26 - len(dd.feature_metadata)) < 5)
        self.assertEqual(len(dd.feature_metadata), 0)

    def test_correlation_complex_spearman(self):
        # test on real complex dataset (timeseries) with spearman correlation
        dd = self.complex.correlation('MF_SAMPLE_NUMBER', method='spearman', random_seed=2017)
        # print(len(dd.feature_metadata))
        self.assertTrue(np.abs(51 - len(dd.feature_metadata)) < 5)

    def test_diff_abundance_kw(self):
        dd = self.test1.diff_abundance_kw(field='group', random_seed=2017)
        expected_ids = [0, 1, 2, 3, 4, 7, 10]
        self.assertEqual(len(dd.feature_metadata), 7)
        for cid in expected_ids:
            self.assertIn(self.test1.feature_metadata.index[cid], dd.feature_metadata.index)

    def test_diff_abundance_paired(self):
        # Do the paired test (we should get 4 features)
        dd = diff_abundance_paired(self.test_paired, 'subj', field='group', val1=1, val2=2, alpha=0.1, random_seed=2020)
        self.assertEqual(len(dd.feature_metadata), 4)
        expected_ids = ['AG', 'AA', 'TA', 'TT']
        for cid in expected_ids:
            self.assertIn(cid, dd.feature_metadata._feature_id)
        # and the unpaired test (which is less sensitive for this dataset - we should get only 2)
        dd = diff_abundance(self.test_paired, field='group', val1=1, val2=2, alpha=0.1, random_seed=2020)
        self.assertEqual(len(dd.feature_metadata), 2)
        expected_ids = ['AG', 'TT']
        for cid in expected_ids:
            self.assertIn(cid, dd.feature_metadata._feature_id)
        # test with binary transforming the pairs
        dd = diff_abundance_paired(self.test_paired, 'subj', transform='pair_rank', field='group', val1=1, val2=2, alpha=0.1, random_seed=2020)
        self.assertEqual(len(dd.feature_metadata), 4)
        expected_ids = ['AA', 'AG', 'TA', 'TT']
        for cid in expected_ids:
            self.assertIn(cid, dd.feature_metadata._feature_id)
        # test with more than 2 samples per group
        dd = diff_abundance_paired(self.test_paired, 'group3', field='group2', val1=1, val2=2, alpha=0.1, random_seed=2020, numperm=1000)
        self.assertEqual(len(dd.feature_metadata), 4)
        expected_ids = ['AA', 'AT', 'GG', 'TA']
        for cid in expected_ids:
            self.assertIn(cid, dd.feature_metadata._feature_id)

    def test_correlation_paired(self):
        # the paired test should find 4 features
        dd = correlation_paired(self.test_paired, 'subj', field='group', alpha=0.1, random_seed=2020)
        self.assertCountEqual(dd.feature_metadata._feature_id, ['AA', 'AG', 'TA', 'TT'])
        self.assertEqual(dd.shape[0], 16)
        # direction is according to the sign of the correlation
        self.assertEqual(dd.feature_metadata.loc['AG', '_calour_direction'], 'group')
        self.assertEqual(dd.feature_metadata.loc['TT', '_calour_direction'], 'Anti-group')
        # and the unpaired test (which is less sensitive for this dataset - we should get only 2)
        dd = self.test_paired.correlation('group', alpha=0.1, random_seed=2020)
        self.assertCountEqual(dd.feature_metadata._feature_id, ['AG', 'TT'])
        # test with ranking within the pairs, and as a class method
        dd = self.test_paired.correlation_paired('subj', field='group', transform='pair_rank', alpha=0.1, random_seed=2020)
        self.assertCountEqual(dd.feature_metadata._feature_id, ['AA', 'AG', 'TA', 'TT'])
        # test with more than 2 samples per group (group3==1 has only 1 sample so it is dropped)
        dd = correlation_paired(self.test_paired, 'group3', field='id', alpha=0.1, random_seed=2020)
        self.assertEqual(dd.shape[0], 15)
        self.assertNotIn('S1', dd.sample_metadata.index)

    def test_correlation_paired_nan_labels(self):
        # samples with non-numeric field values are dropped before pairing,
        # so their pair partner becomes a singleton and is dropped too
        exp = self.test_paired.copy()
        exp.sample_metadata['group'] = exp.sample_metadata['group'].astype(object)
        exp.sample_metadata.loc['S1', 'group'] = 'na'
        dd = correlation_paired(exp, 'subj', field='group', alpha=0.1, random_seed=2020)
        self.assertEqual(dd.shape[0], 14)
        self.assertNotIn('S1', dd.sample_metadata.index)
        self.assertNotIn('S2', dd.sample_metadata.index)

    def test_correlation_paired_shuffler(self):
        # the paired shuffler should only permute labels within each pair_field value
        exp, transform, shuffler = _prepare_paired(self.test_paired, 'group3', 'rankdata', random_seed=2020)
        self.assertEqual(transform, 'rankdata')
        labels = exp.sample_metadata['id'].values.astype(float)
        groups = exp.sample_metadata['group3'].values
        changed = False
        for i in range(10):
            shuffled = shuffler(labels)
            for cgroup in np.unique(groups):
                self.assertCountEqual(shuffled[groups == cgroup], labels[groups == cgroup])
            changed = changed or not np.array_equal(shuffled, labels)
        self.assertTrue(changed)

    def test_correlation_paired_bad_field(self):
        with self.assertRaises(ValueError):
            correlation_paired(self.test_paired, 'subj', field='not_a_field')
        with self.assertRaises(ValueError):
            correlation_paired(self.test_paired, 'not_a_field', field='group')


if __name__ == "__main__":
    unittest.main()

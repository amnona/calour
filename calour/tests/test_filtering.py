# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from unittest import TestCase, main

from skbio.util import get_data_path

from calour.util import Tests
import calour as ca


class FilteringTests(Tests):
    def setUp(self):
        super().setUp()
        self.simple = ca.read(self.simple_table, self.simple_map)

    def test_filter_by_metadata_sample(self):
        obs = self.simple.filter_by_metadata('group', '1')
        self.assertIsNot(obs, self.simple)

        exp = ca.read(*[get_data_path(i) for i in ['filter.1.biom', 'filter.1_sample.txt', 'filter.1_feature.txt']])
        self.assertEqual(obs, exp)

        obs = self.simple.filter_by_metadata('group', '1', inplace=True)
        self.assertIs(obs, self.simple)


    def test_filter_by_metadata_feature(self)
        obs = self.simple.filter_by_metadata('taxonomy', 'bad_bacteria', axis=1)
        self.assertIsNot(obs, self.simple)

        exp = ca.read(*[get_data_path(i) for i in ['filter.2.biom', 'filter.2_sample.txt', 'filter.2_feature.txt']])
        self.assertEqual(obs, exp)

        obs = self.simple.filter_by_metadata('taxonomy', 'bad_bacteria', inplace=True)
        self.assertIs(obs, self.simple)

    def test_filter_by_data(self):
        obs = self.simple.filter_by_data('sum_abundance')


if __name__ == '__main__':
    main()

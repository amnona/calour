# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from unittest import main
from copy import deepcopy
from os.path import basename, join
from tempfile import mkdtemp
import shutil
import warnings

import calour as ca
from calour import util
import numpy as np
import pandas as pd

from calour._testing import Tests


class IOTests(Tests):
    def setUp(self):
        super().setUp()
        self.test1 = ca.read(self.test1_biom, self.test1_samp, normalize=None)

    def test_get_taxonomy_string(self):
        orig_tax = list(self.test1.feature_metadata['taxonomy'].values)
        # test string taxonomy
        tax = util._get_taxonomy_string(self.test1)
        self.assertListEqual(tax, orig_tax)
        # test using a biom table with list taxonomy, not removing the X__ parts
        exp = deepcopy(self.test1)
        exp.feature_metadata['taxonomy'] = exp.feature_metadata['taxonomy'].str.split(';')
        tax = util._get_taxonomy_string(exp, remove_underscore=False)
        self.assertListEqual(tax, orig_tax)
        # and test with removing the X__ parts and lower case
        tax = util._get_taxonomy_string(exp, to_lower=True)
        self.assertEqual(tax[1], 'bacteria;tenericutes;mollicutes;mycoplasmatales;mycoplasmataceae;mycoplasma;')

    def test_get_file_md5(self):
        md5 = util.get_file_md5(self.test1_samp)
        self.assertEqual(md5, '36c9dc4dd389e82689a2a53ca9558c6b')

    def test_get_data_md5(self):
        exp = deepcopy(self.test1)
        # try on dense matrix
        exp.sparse = True
        md5 = util.get_data_md5(exp.data)
        self.assertEqual(md5, '561ba229f4a4c68979e560a10cc3fe42')

        # try on sparse matrix
        exp.sparse = False
        md5 = util.get_data_md5(exp.data)
        self.assertEqual(md5, '561ba229f4a4c68979e560a10cc3fe42')

    def test_get_config_file(self):
        fp = util.get_config_file()
        self.assertTrue(basename(fp).startswith('calour.config'))

    def test_get_config_sections(self):
        sections = util.get_config_sections()
        self.assertIn('dbbact', sections)
        self.assertIn('sponge', sections)
        self.assertNotIn('username', sections)

    def test_config_file_value(self):
        # test the set and get config file values
        # create the tmp config file path
        d = mkdtemp()
        f = join(d, 'config.txt')
        util.set_config_value('test1', 'val1', config_file_name=f)
        res = util.get_config_value('test1', config_file_name=f)
        self.assertEqual(res, 'val1')
        # test the fallback if a key doesn't exist
        res = util.get_config_value('test2', fallback='na', config_file_name=f)
        self.assertEqual(res, 'na')
        shutil.rmtree(d)

    def test_to_list(self):
        self.assertEqual(util._to_list(5), [5])
        self.assertEqual(util._to_list([5]), [5])
        self.assertEqual(util._to_list('test'), ['test'])
        self.assertEqual(util._to_list(range(5)), range(5))

    def test_format_docstring(self):
        @util.format_docstring([], a=1, b='a')
        def foo():
            '''{} {a} {b}'''
        self.assertEqual(foo.__doc__, '[] 1 a')

    def test_join_fields(self):
        df = pd.DataFrame({'a': ['dog', 'monkey'], 'b': ['bone', 'banana']})

        res = util.join_fields(df.copy(), 'a', 'b')
        self.assertListEqual(list(res['a_b']), ['dog_bone', 'monkey_banana'])

        res = util.join_fields(df.copy(), 'a', 'b', new_field='joined', sep='|', pad='-')
        self.assertIn('joined', res.columns)
        self.assertListEqual(list(res['joined']), ['dog---|--bone', 'monkey|banana'])

    def test_compute_prevalence(self):
        cutoffs, prevalences = util.compute_prevalence([0, 1, 0, 2, 4])
        np.testing.assert_array_equal(cutoffs, np.array([0, 1, 2, 4]))
        np.testing.assert_allclose(prevalences, np.array([0.6, 0.4, 0.2, 0.0]))

    def test_transition_index(self):
        transitions = list(util._transition_index(['a', 'a', 'b', 1, 2, None, None]))
        self.assertEqual(transitions, [(2, 'a'), (3, 'b'), (4, 1), (5, 2), (7, None)])

    def test_convert_axis_name(self):
        @util._convert_axis_name
        def _get_axis(axis=0):
            return axis

        self.assertEqual(_get_axis(axis='sample'), 0)
        self.assertEqual(_get_axis(axis='f'), 1)
        self.assertEqual(_get_axis(axis=1), 1)
        with self.assertRaises(KeyError):
            _get_axis(axis='bad_axis')

    def test_get_file_md5_none(self):
        self.assertIsNone(util.get_file_md5(None))

    def test_get_dataframe_md5(self):
        df1 = pd.DataFrame({'x': [1, 2], 'y': [['a'], ['b']]})
        df2 = pd.DataFrame({'x': [1, 2], 'y': [['a'], ['b']]})
        df3 = pd.DataFrame({'x': [1, 3], 'y': [['a'], ['b']]})

        md5_1 = util.get_dataframe_md5(df1)
        md5_2 = util.get_dataframe_md5(df2)
        md5_3 = util.get_dataframe_md5(df3)
        self.assertEqual(md5_1, md5_2)
        self.assertNotEqual(md5_1, md5_3)
        self.assertIsNone(util.get_dataframe_md5(None))

    def test_argsort(self):
        vals = [10, 'b', np.nan, 2.5, 'a']
        idx = util._argsort(vals)
        self.assertEqual(idx, [3, 0, 2, 4, 1])
        idx_rev = util._argsort(vals, reverse=True)
        self.assertEqual(idx_rev, [1, 4, 2, 0, 3])

    def test_deprecated(self):
        @util.deprecated('use new_func instead')
        def old_func(x):
            return x + 1

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            self.assertEqual(old_func(4), 5)
            self.assertEqual(len(w), 1)
            self.assertIn('deprecated function', str(w[0].message))


if __name__ == "__main__":
    main()

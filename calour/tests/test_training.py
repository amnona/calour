# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from unittest import main
from os.path import join
import logging

from numpy.testing import assert_array_equal, assert_almost_equal
import numpy as np
import pandas as pd
import pandas.testing as pdt
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import KFold

import calour as ca
from calour._testing import Tests
from calour.training import (
    plot_cm, plot_roc, plot_prc, plot_scatter,
    SortedStratifiedKFold, RepeatedSortedStratifiedKFold,
    _interpolate_precision_recall)


class TTests(Tests):
    def setUp(self):
        super().setUp()
        self.test2_sparse = ca.read(self.test2_biom, self.test2_samp, self.test2_feat, normalize=None)
        self.test2_dense = ca.read(self.test2_biom, self.test2_samp, self.test2_feat, sparse=False, normalize=None)

    def test_add_sample_metadata_as_features(self):
        new = self.test2_sparse.add_sample_metadata_as_features(['categorical'])
        dat = new.data.toarray()
        assert_array_equal(dat[:, 0:3],
                           [[1, 0, 0], [0, 1, 0], [0, 0, 1]] * 3)
        self.assertListEqual(new.feature_metadata.index[:3].tolist(),
                             ['categorical=A', 'categorical=B', 'categorical=C'])

    def test_add_sample_metadata_as_features_dense(self):
        new = self.test2_dense.add_sample_metadata_as_features(['categorical'])
        assert_array_equal(new.data[:, 0:3],
                           [[1, 0, 0], [0, 1, 0], [0, 0, 1]] * 3)
        self.assertListEqual(new.feature_metadata.index[:3].tolist(),
                             ['categorical=A', 'categorical=B', 'categorical=C'])

    def test_split_train_test(self):
        train, test = self.test2_dense.split_train_test(
            test_size=3, stratify='categorical', shuffle=True, random_state=7)

        self.assert_experiment_equal(
            test, self.test2_dense.filter_ids(['S3', 'S8', 'S1'], axis='s'))
        self.assert_experiment_equal(
            train, self.test2_dense.filter_ids(['S9', 'S6', 'S5', 'S2', 'S4', 'S7'], axis='s'))

    def test_regress(self):
        diabetes = datasets.load_diabetes()
        X = diabetes.data[:9]
        y = diabetes.target[:9]
        smd = pd.DataFrame({'diabetes': y})
        exp = ca.Experiment(X, smd, sparse=False)
        run = exp.regress('diabetes', KNeighborsRegressor(), KFold(3, shuffle=True, random_state=0))
        observed = next(run)
        expected = pd.read_table(join(self.test_data_dir, 'test_regress.txt'), index_col=0)

        # make sure the column order are the same for comparison
        pdt.assert_frame_equal(observed.sort_index(axis=1), expected.sort_index(axis=1))

    def test_plot_scatter(self):
        res = pd.read_table(join(self.test_data_dir, 'diabetes_pred.txt'), index_col=0)
        title = 'foo'
        ax = plot_scatter(res, title=title)
        self.assertEqual(title, ax.get_title())
        cor = 'r=-0.62 p-value=0.078'
        self.assertEqual(cor, ax.texts[0].get_text())
        dots = []
        for collection in ax.collections:
            dots.append(collection.get_offsets())
        assert_array_equal(np.concatenate(dots, axis=0),
                           res[['Y_TRUE', 'Y_PRED']].values)

    def test_classify(self):
        iris = datasets.load_iris()
        n = len(iris.target)
        np.random.seed(0)
        i = np.random.randint(0, n, 36)
        X = iris.data[i]
        y = iris.target[i]
        d = dict(enumerate(iris.target_names))
        smd = pd.DataFrame({'plant': y}).replace(d)
        exp = ca.Experiment(X, smd, sparse=False)
        run = exp.classify('plant', KNeighborsClassifier(),
                           predict='predict_proba',
                           cv=KFold(3, shuffle=True, random_state=0))
        observed = next(run)
        expected = pd.read_table(join(self.test_data_dir, 'test_classify.txt'), index_col=0)
        pdt.assert_frame_equal(expected, observed)
        # plot_roc(observed)
        # from matplotlib import pyplot as plt
        # plt.show()

    def test_plot_roc_multi(self):
        result = pd.read_table(join(self.test_data_dir, 'iris_pred.txt'))
        ax, _ = plot_roc(result)
        legend = ax.get_legend()
        exp = {'Luck',
               'setosa (0.99 $\\pm$ 0.00)',
               'virginica (0.96 $\\pm$ 0.05)',
               'versicolor (0.95 $\\pm$ 0.07)'}
        obs = {i.get_text() for i in legend.get_texts()}
        self.assertSetEqual(exp, obs)
        # from matplotlib import pyplot as plt
        # plt.show()

    def test_plot_roc_multi_no_cv(self):
        result = pd.read_table(join(self.test_data_dir, 'iris_pred.txt'))
        ax, _ = plot_roc(result, cv=False)
        legend = ax.get_legend()
        exp = {'Luck',
               'setosa (1.00)',
               'virginica (0.94)',
               'versicolor (0.92)'}
        obs = {i.get_text() for i in legend.get_texts()}
        self.assertSetEqual(exp, obs)

    def test_plot_roc_binary(self):
        result = pd.read_table(join(self.test_data_dir, 'iris_pred.txt'))
        result['Y_TRUE'] = ['virginica' if i == 'virginica' else 'not virginica'
                            for i in result['Y_TRUE']]
        result['not virginica'] = 1 - result['virginica']
        ax, _ = plot_roc(result, classes=['virginica'])
        # from matplotlib import pyplot as plt
        # plt.show()
        legend = ax.get_legend()
        exp = {'Luck',
               'virginica (0.96 $\\pm$ 0.05)'}
        obs = {i.get_text() for i in legend.get_texts()}
        self.assertSetEqual(exp, obs)

    def test_plot_roc_warning(self):
        prob = np.arange(0, 1, 0.1)
        result = pd.DataFrame({'pos': 1 - prob,
                               'neg': prob,
                               'Y_TRUE': ['pos'] * 9 + ['neg'],
                               'CV': [0, 1] * 5})
        # re-enable logging because it is disabled in the parent setUp
        logging.disable(logging.NOTSET)
        # test for calour printed warning message
        with self.assertLogs(level='WARNING') as cm:
            # test (and capture) scikit-learn printed warning message
            with self.assertWarnsRegex(UserWarning, 'No positive samples in y_true, true positive value should be meaningless'):
                plot_roc(result)
                self.assertRegex(cm.output[0], 'no true positive or no negative samples')

    def test_interpolate_precision_recall(self):
        n = 9
        recall = np.linspace(0.0, 1.0, num=n)
        rand = np.random.RandomState(9)
        precision = rand.rand(n) * (1 - recall)

        x = np.linspace(0, 1, num=20)
        obs = _interpolate_precision_recall(x, recall, precision)
        exp = np.array([0.43914, 0.43914, 0.43914, 0.37183, 0.37183, 0.104627,
                        0.104627, 0.104627, 0.104627, 0.104627, 0.104627, 0.104627,
                        0.104627, 0.104627, 0.104627, 0.031013, 0.031013, 0.,
                        0., 0.])
        assert_almost_equal(obs, exp, decimal=5)
        # # use the plot to visually check the func works as expected
        # from matplotlib import pyplot as plt
        # fig, axes = plt.subplots(nrows=2, ncols=1)
        # axes[0].hold(True)
        # axes[0].plot(recall, precision, '--b')
        # decreasing_max_precision = np.maximum.accumulate(precision[::-1])[::-1]
        # axes[0].step(recall, decreasing_max_precision, '-r')
        # axes[1].step(x, obs, '-g')
        # plt.show()

    def test_plot_prc(self):
        # generating test data set:
        # http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#plot-the-precision-recall-curve
        # from sklearn import svm, datasets
        # from sklearn.model_selection import train_test_split
        # iris = datasets.load_iris()
        # X = iris.data
        # y = iris.target
        # # Add noisy features
        # random_state = np.random.RandomState(0)
        # n_samples, n_features = X.shape
        # X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

        # # Limit to the two first classes, and split into training and test
        # X_train, X_test, y_train, y_test = train_test_split(X[y < 2], y[y < 2],
        #                                                     test_size=.5,
        #                                                     random_state=random_state)
        # # Create a simple classifier
        # classifier = svm.LinearSVC(random_state=random_state)
        # classifier.fit(X_train, y_train)
        # y_score = classifier.decision_function(X_test)
        # result = pd.DataFrame({'Y_TRUE': ['setosa' if i == 1 else 'non-setosa' for i in y_test],
        #                        'setosa': y_score, 'CV': 0})

        f = join(self.test_data_dir, 'plot_prc.txt')
        result = pd.read_table(f, index_col=0)
        ax = plot_prc(result, classes=['setosa'])
        legend = ax.get_legend()
        exp = {'iso-f1 curves',
               'setosa (0.88 $\\pm$ 0.00)'}
        obs = {i.get_text() for i in legend.get_texts()}
        self.assertSetEqual(exp, obs)

    def test_plot_cm(self):
        result = pd.read_table(join(self.test_data_dir, 'iris_pred.txt'), index_col=0)
        ax = plot_cm(result, classes=['setosa', 'virginica', 'versicolor'])
        # from matplotlib import pyplot as plt
        # plt.show()

        # make sure you don't confuse x, y label
        self.assertEqual(ax.get_xlabel(), 'Prediction')
        self.assertEqual(ax.get_ylabel(), 'Observation')

        obs = [((0, 0), '13'), ((1, 0), '0'), ((2, 0), '0'),
               ((0, 1), '0'), ((1, 1), '9'), ((2, 1), '1'),
               ((0, 2), '0'), ((1, 2), '3'), ((2, 2), '10')]
        for exp, obs in zip(ax.texts, obs):
            self.assertEqual(exp.get_text(), obs[1])
            self.assertEqual(exp.get_position(), obs[0])


class RTests(Tests):
    def setUp(self):
        self.y = np.array([9.1, 7.1, 8.1, 5.1, 3.1, 1.1, 2.1, 6.1, 4.1])
        self.X = self.y[:, np.newaxis]

    def test_sorted_stratified(self):
        n = self.y.shape[0]
        for k in (2, 3, 4):
            ssk = SortedStratifiedKFold(k, shuffle=True)
            for train, test in ssk.split(self.X, self.y):
                # check the size of the test fold
                ni = int(n / k)
                self.assertTrue(test.shape[0] == ni or test.shape[0] == ni + 1)

                # check every data point is either in train or fold and only once
                idx = np.concatenate([train, test])
                idx.sort()
                assert_array_equal(idx, np.arange(n))

                # print(train, test)
                # print(np.sort(self.y[train]), np.sort(self.y[test]))

                # check there is a value in each bin in the test fold
                y_test = self.y[test]
                for i in range(1, ni + 1):
                    cutoff = i * k + 0.1
                    self.assertEqual(np.sum(y_test <= cutoff), i)

    def test_rep_sorted_stratified(self):
        n = self.y.shape[0]
        for k in (3, 2):
            ssk = RepeatedSortedStratifiedKFold(n_splits=k, n_repeats=2)
            for train, test in ssk.split(self.X, self.y):
                # check the size of the test fold
                ni = int(n / k)
                self.assertTrue(test.shape[0] == ni or test.shape[0] == ni + 1)

                # check every data point is either in train or fold and only once
                idx = np.concatenate([train, test])
                idx.sort()
                assert_array_equal(idx, np.arange(n))

                # check there is a value in each bin in the test fold
                y_test = self.y[test]
                for i in range(1, ni + 1):
                    cutoff = i * k + 0.1
                    self.assertEqual(np.sum(y_test <= cutoff), i)


if __name__ == "__main__":
    main()

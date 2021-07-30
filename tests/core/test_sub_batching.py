import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from ism.core.sub_batching import sub_batch, unroll_predictions


def test_sub_batching():
    X = np.array([
        [100.0],
        [200.0],
        [300.0],
        [400.0],
        [500.0],
    ])

    assert_array_equal(
        x=np.array([
            [[100.0]],
            [[200.0]],
            [[300.0]],
            [[400.0]],
            [[500.0]],
        ]),
        y=sub_batch(X, sub_batch_size=1)
    )

    assert_array_equal(
        x=np.array([
            [[100.0], [200.0]],
            [[200.0], [300.0]],
            [[300.0], [400.0]],
            [[400.0], [500.0]],
        ]),
        y=sub_batch(X, sub_batch_size=2)
    )

    assert_array_equal(
        x=np.array([
            [[100.0], [200.0], [300.0]],
            [[200.0], [300.0], [400.0]],
            [[300.0], [400.0], [500.0]],
        ]),
        y=sub_batch(X, sub_batch_size=3)
    )


def test_sub_batching_multi_dimensional_data():
    X = np.array([
        [1.0, 10.0, 100.0],
        [2.0, 20.0, 200.0],
        [3.0, 30.0, 300.0],
        [4.0, 40.0, 400.0],
        [5.0, 50.0, 500.0],
    ])

    assert_array_equal(
        x=np.array([
            [[1.0, 10.0, 100.0], [2.0, 20.0, 200.0]],
            [[2.0, 20.0, 200.0], [3.0, 30.0, 300.0]],
            [[3.0, 30.0, 300.0], [4.0, 40.0, 400.0]],
            [[4.0, 40.0, 400.0], [5.0, 50.0, 500.0]],
        ]),
        y=sub_batch(X, sub_batch_size=2)
    )
    assert_array_equal(
        x=np.array([
            [[1.0, 10.0, 100.0], [2.0, 20.0, 200.0], [3.0, 30.0, 300.0]],
            [[2.0, 20.0, 200.0], [3.0, 30.0, 300.0], [4.0, 40.0, 400.0]],
            [[3.0, 30.0, 300.0], [4.0, 40.0, 400.0], [5.0, 50.0, 500.0]],
        ]),
        y=sub_batch(X, sub_batch_size=3)
    )


def test_unrrol_predictions():
    y_pred = np.array([
        [[1.0],
         [2.1],
         [2.9]],

        [[1.9],
         [3.0],
         [4.0]],

        [[3.1],
         [4.0],
         [5.0]],

        [[4.0],
         [5.0],
         [6.0]],

        [[5.0],
         [6.0],
         [7.0]]
    ])

    assert_array_equal(
        x=np.array([
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
            [6.0],
            [7.0]
        ]),
        y=unroll_predictions(y_pred)
    )


def test_unrrol_predictions_multi_dimensional_data():
    y_pred = np.array([
        [[0.90, 0.10],
         [0.79, 0.21],
         [0.70, 0.30]],

        [[0.81, 0.19],
         [0.71, 0.29],
         [0.60, 0.40]],

        [[0.69, 0.31],
         [0.61, 0.39],
         [0.50, 0.50]],

        [[0.59, 0.41],
         [0.51, 0.49],
         [0.39, 0.61]],

        [[0.49, 0.51],
         [0.41, 0.59],
         [0.30, 0.70]]
    ])

    assert_array_almost_equal(
        x=np.array([
            [0.9, 0.1],
            [0.8, 0.2],
            [0.7, 0.3],
            [0.6, 0.4],
            [0.5, 0.5],
            [0.4, 0.6],
            [0.3, 0.7]
        ]),
        y=unroll_predictions(y_pred)
    )

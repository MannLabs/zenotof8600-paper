import numpy as np

from zenotof8600_analytics.ztscan.profile import peak_index


def test_symmetric_peak_is_on_the_sample():
    assert peak_index(np.array([[0.0, 1.0, 3.0, 1.0, 0.0]]))[0] == 2.0


def test_parabola_vertex_between_samples():
    x = np.arange(7.0)
    y = -(x - 3.3) ** 2 + 10
    assert np.isclose(peak_index(y[None, :])[0], 3.3)


def test_maximum_at_the_edge_is_not_extrapolated():
    assert peak_index(np.array([[5.0, 3.0, 1.0], [1.0, 3.0, 5.0]])).tolist() == [0.0, 2.0]

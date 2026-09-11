from identifiability import singular_values_at_g, singular_values_sweep, near_degenerate_clusters
import numpy as np

gs_real = np.arange(0.1, 4, 0.1).tolist()

def test_singular_values_at_g1():
    for g_value in gs_real:
        assert len(singular_values_at_g(g_value)) == 10

def test_singular_values_at_g2():
    for g_value in gs_real:
        assert np.all(singular_values_at_g(g_value)[:-1] >= singular_values_at_g(g_value)[1:])

def test_singular_values_at_g3():
    for g_value in gs_real:
        assert np.all(singular_values_at_g(g_value) >= 0)

def test_singular_values_sweep1():
        assert singular_values_sweep(gs_real).shape == (len(gs_real), 10)

def test_singular_values_sweep2():
    for g_value, idx in zip(gs_real, range(len(gs_real))):
        assert np.all(singular_values_at_g(g_value) == singular_values_sweep(gs_real)[idx])

def test_singular_values_sweep3():
    for g_value in gs_real:
        assert singular_values_sweep([g_value]).shape == (1,10)

def test_near_degenerate_clusters1():
    assert [[tup[1] for tup in sublist] for sublist in near_degenerate_clusters(0.01, 0.1)] == [[1], [2, 3], [4, 5, 6, 7, 8, 9], [10]]

def test_near_degenerate_clusters2():
    assert [[tup[1] for tup in sublist] for sublist in near_degenerate_clusters(0.05, 0.1)] == [[1], [2, 3], [4, 5, 6, 7, 8, 9], [10]]

def test_near_degenerate_clusters3():
    for g_value in gs_real:
        flat = [item[1] for sublist in near_degenerate_clusters(g_value, 0.1) for item in sublist]
        assert set(flat) == set(range(1, 11))
        assert len(flat) == 10

def test_near_degenerate_clusters4():
    assert [[tup[1] for tup in sublist] for sublist in near_degenerate_clusters(0.01, 100)] == [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]

def test_near_degenerate_clusters5():
    assert [[tup[1] for tup in sublist] for sublist in near_degenerate_clusters(0.01, 1e-10)] == [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
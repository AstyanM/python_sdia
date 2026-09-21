import pytest
import numpy as np
import sys
import os

# Add the parent directory (which contains src) to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gradient import gradient2D, tv, gradient2D_adjoint

def test_gradient2D():
    M, N = 3, 3
    X_const = np.ones((M, N))
    D_const = gradient2D(X_const)
    assert D_const.shape == (2, M, N)
    assert np.allclose(D_const, 0)
    
    X_val = np.array([[1, 2], [3, 4], [5, 6]])
    D_val = gradient2D(X_val)
    expected_X_dh = np.array([[1, 0], [1, 0], [1, 0]])
    expected_D_vx = np.array([[2, 2], [2, 2], [0, 0]])
    assert np.allclose(D_val[0], expected_X_dh)
    assert np.allclose(D_val[1], expected_D_vx)
    print("\ntest_gradient2D passed")

def test_tv():
    X = np.array([[1, 2], [3, 4]])
    expected_tv = 3 + np.sqrt(5)
    assert np.isclose(tv(X), expected_tv)
    print("\ntest_tv passed")

def test_adjoint():
    np.random.seed(42)
    M, N = 10, 12
    X = np.random.randn(M, N) + 1j * np.random.randn(M, N)
    Y = np.random.randn(2, M, N) + 1j * np.random.randn(2, M, N)
    
    DX = gradient2D(X)
    DsY = gradient2D_adjoint(Y)
    
    inner1 = np.sum(np.conj(DX) * Y)
    inner2 = np.sum(np.conj(X) * DsY)
    
    assert np.isclose(inner1, inner2)
    print("\ntest_adjoint passed")
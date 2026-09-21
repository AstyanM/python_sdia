import numpy as np

def gradient2D(X):
    """
    Computes the 2D discrete gradient operator D applied to a matrix X.
    
    Parameters:
    X : array_like, 2D
        Input matrix of shape (M, N).
        
    Returns:
    D : ndarray
        Output tensor of shape (2, M, N), where D[0] is X*D_h and D[1] is D_v*X.
    """
    assert X.ndim == 2, "Input array must have exactly 2 dimensions"
    
    # Horizontal differences: append a column of zeros
    X_dh = np.c_[np.diff(X, axis=1), np.zeros(X.shape[0])]
    
    # Vertical differences: append a row of zeros
    D_vx = np.r_[np.diff(X, axis=0), np.zeros((1, X.shape[1]))]
    
    # Combine into a single tensor of shape (2, M, N)
    return np.array([X_dh, D_vx])


def tv(X):
    """
    Computes the discrete isotropic total variation of a matrix X.
    
    Parameters:
    X : array_like, 2D
        Input matrix of shape (M, N).
        
    Returns:
    tv_val : float
        Total variation of X.
    """
    D = gradient2D(X)
    return np.sum(np.sqrt(np.abs(D[0])**2 + np.abs(D[1])**2))


def gradient2D_adjoint(Y):
    """
    Computes the adjoint of the 2D discrete gradient operator applied to Y.
    Y : array_like, shape (2, M, N)
    """
    Y_h, Y_v = Y[0], Y[1]
    M, N = Y_h.shape
    
    # Horizontal part (adjoint of diff + pad)
    if N > 1:
        diff_h = Y_h[:, 1:N-1] - Y_h[:, 0:N-2]
        Yh_Dh = np.c_[-Y_h[:, 0], -diff_h, Y_h[:, N-2]]
    else:
        Yh_Dh = np.zeros_like(Y_h)
        
    # Vertical part (adjoint of diff + pad)
    if M > 1:
        diff_v = Y_v[1:M-1, :] - Y_v[0:M-2, :]
        Dv_Yv = np.r_[[-Y_v[0, :]], -diff_v, [Y_v[M-2, :]]]
    else:
        Dv_Yv = np.zeros_like(Y_v)
        
    return Yh_Dh + Dv_Yv


def gradientND(X):
    """
    Computes the ND discrete gradient operator D applied to an ND array X.
    Returns a tensor of shape (p, N_1, ..., N_p) where p = X.ndim.
    """
    p = X.ndim
    grads = []
    for axis in range(p):
        diff = np.diff(X, axis=axis)
        # Create an array of zeros with the same shape as X, except dimension 'axis' is 1
        zero_shape = list(X.shape)
        zero_shape[axis] = 1
        zeros = np.zeros(zero_shape, dtype=X.dtype)
        # Concatenate zeros to match the original shape
        grad_axis = np.concatenate([diff, zeros], axis=axis)
        grads.append(grad_axis)
    return np.array(grads)


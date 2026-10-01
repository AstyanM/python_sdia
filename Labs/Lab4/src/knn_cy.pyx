import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=1] knn_classifier_cy(
        np.float64_t[:, :] x_train, 
        np.float64_t[:] y_train, 
        np.float64_t[:, :] x_test, 
        int k):

    cdef int N_train = x_train.shape[0]
    cdef int N_test = x_test.shape[0]
    cdef int n_features = x_train.shape[1]

    cdef np.ndarray[np.float64_t, ndim=1] class_pred = np.zeros(N_test, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] distances = np.zeros(N_train, dtype=np.float64)

    cdef double dist, diff
    cdef int i, j, f

    for i in range(N_test):
        for j in range(N_train):
            dist = 0.0
            for f in range(n_features):
                diff = x_train[j, f] - x_test[i, f]
                dist += diff * diff
            distances[j] = sqrt(dist)

        # Using Python for sorting and bincount for simplicity
        dist_np = np.asarray(distances)
        id_k = np.argpartition(dist_np, k)[:k]

        labels_k = np.asarray(y_train)[id_k]
        counts = np.bincount(labels_k.astype(int))
        class_pred[i] = np.argmax(counts)

    return class_pred

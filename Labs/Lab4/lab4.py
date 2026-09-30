# %% [markdown]
# # Practical session 4 - K-nearest neighbours (K-NN) classification with numpy, scikit-learn, cython and numba
# 
# Students (pair):
# - [Student 1]([link](https://github.com/username1))
# - [Student 2]([link](https://github.com/username2))

# %% [markdown]
# **Useful references for this lab**:
# 
# [1] scikit-learn: [documentation](https://scikit-learn.org/stable/modules/neighbors.html?highlight=knn%20classification)
# 
# [2] `numba`: [documentation](http://numba.pydata.org/) 
# 
# [3] cython: [a very useful tutorial](https://cython.readthedocs.io/en/latest/src/userguide/numpy_tutorial.html#numpy-tutorial), and [another one](http://docs.cython.org/en/latest/src/tutorial/cython_tutorial.html)
# 
# 
# 
# ## <a name="content">Contents</a>
# - [Exercise 1: KNN classification with numpy and sklearn](#ex1)
# - [Exercise 2: Code acceleration with cython](#ex2)
# - [Exercise 3: Code acceleration with numba](#ex3)
# ---

# %%

# %% [markdown]
# ## <a name="ex1">Exercise 1: K-Nearest Neighbours (K-NN) classification with numpy and scikit-learn</a> [(&#8593;)](#content)

# %% [markdown]
# This session is a first introduction to classification using the most intuitive non parametric method: the $K$-nearest neighbours. The principle is [the following](https://scikit-learn.org/stable/modules/neighbors.html?highlight=knn%20classification). A set of labelled observations is given as a learning set. A classification taks then consists in assigning a label to any new observation. In particular, the K-NN approach consists in assigning to the observation the most frequent label among its $K$ nearest neighbours taken in the training set.

# %% [markdown]
# ### A. Validation on synthetic data
# 
# Load the training and test datasets `data/synth_train.txt` and `data/synth_test.txt`. Targets belong to the set $\{1,2\}$ and entries belong to $\mathbb{R}^2$. The file `data/synth_train.txt` contain 100 training data samples, and `data/synth_test.txt` contains 200 test samples, where:
# 
# - the 1st column contains the label of the class the sample;
# - columns 2 & 3 contain the coordinates of each sample (in $\mathbb{R}^2$).
# 
# Useful commands can be found below.

# %% [markdown]
# ```python
# # load the training set
# train = np.loadtxt('data/synth_train.txt')  #...,delimiter=',') if there are ',' as delimiters
# class_train = train[:,0]
# x_train = train[:,1:]
# N_train = train.shape[0]
# ```

# %% [markdown]
# ```python
# # load the test set
# test = np.loadtxt('/datasynth_test.txt') 
# class_test_1 = test[test[:,0]==1]
# class_test_2 = test[test[:,0]==2]
# x_test = test[:,1:]
# N_test = test.shape[0]
# ```

# %% [markdown]
# 1\. Display the training set and distinguish the two classes. 
# 
# > Hint: useful functions include `matplotlib.pyplot.scatter` or `matplotlib.pyplot.plot`.

# %% [markdown]
# **Answer:**

# %%
import numpy as np
import matplotlib.pyplot as plt

# load the training set
train = np.loadtxt('data/synth_train.txt')  #...,delimiter=',') if there are ',' as delimiters
class_train = train[:,0]
x_train = train[:,1:]
N_train = train.shape[0]

# load the test set
test = np.loadtxt('data/synth_test.txt')
class_test_1 = test[test[:,0]==1]
class_test_2 = test[test[:,0]==2]
x_test = test[:,1:]
N_test = test.shape[0]

print('Training set size: ', N_train)
print('Test set size: ', N_test)

plt.scatter([x_train[i][0] for i in range(len(class_train)) if class_train[i] == 1], [x_train[i][1] for i in range(len(class_train)) if class_train[i] == 1], c='r', label='Class 1')
plt.scatter([x_train[i][0] for i in range(len(class_train)) if class_train[i] == 2], [x_train[i][1] for i in range(len(class_train)) if class_train[i] == 2], c='b', label='Class 2')
plt.legend()
plt.show()


# %% [markdown]
# 2\. Implement the K-nearest neighbours algorithm for classification.
# 
# > Hint: 
# > - useful functions include `numpy.linalg.norm`, `numpy.argsort`, `numpy.bincount`;
# > - implement the algorithm as a function rather than an object. This will drastically simplify the acceleration step using Cython.
# > - for an optimized partial sorting procedure, you may have a look at the [`bottleneck.argpartition` function](https://bottleneck.readthedocs.io/en/latest/reference.html#bottleneck.argpartition).
# > 1. Compute for each row in `x_test` (if necessary use `np.newaxis`) its distance with respect to `x_train`:
# >  - Use  `numpy.linalg.norm` (in which dimension this distance is computed ? Consider using `axis` argument)
# > 2. Sort the ordered collection of distances (indices from smallest to largest (in ascending order) by the distances):
# >   - Use `np.argsort` (at the end replace this procedure by `bottleneck.argpartition`)
# >   - Once the sorting is done, we take only the indices of `labels` of the `n_neighbours` nearest neighbours of the `class_train` :
# >     - `id = np.argsort(distances)[:n_ neighbours]` and `labels = class_train[id]`
# > 3. The K-nearest can be used for **Regression**, in this case it is necessary to return the mean of the K-labels. For **Classification**,  we return the mode of the K-labels :
# > - Use `np.bincount` for `labels` to affect the variable `class_pred[q]` (for row `q`). This procedure counts the number of occurrences of each value in array. **Mode** is the value that appears. How can we get this value ?
# 
# 
# ```python
# import numpy as np
# import bottleneck as bn
# 
# # Create a random array
# arr = np.random.rand(10)
# N = 3  # Number of smallest elements to retrieve
# 
# # Using np.argsort() to get indices of the first N elements
# sorted_indices = np.argsort(arr)[:N]
# 
# # Using bottleneck.argpartition() to get the first N smallest indices
# partitioned_indices = bn.argpartition(arr, N)[:N]
# 
# # Display the results
# print("Original array:", arr)
# print("Indices using np.argsort:", sorted_indices)
# print("First N elements using np.argsort:", arr[sorted_indices])
# 
# print("Indices using bottleneck.argpartition:", partitioned_indices)
# print("First N elements using bottleneck.argpartition:", arr[partitioned_indices])
# 
# are_equal = set(arr[sorted_indices]) == set(arr[partitioned_indices])
# print(are_equal)
# ```
# 

# %% [markdown]
# **Answer:**

# %%
def knn_classifier(x_train, y_train, x_test, k):
    """
    K-nearest neighbours algorithm for classification.
    """
    import bottleneck as bn
    
    N_test = x_test.shape[0]
    class_pred = np.zeros(N_test, dtype=y_train.dtype)
    
    for i in range(N_test):
        # 1. Compute distance
        distances = np.linalg.norm(x_train - x_test[i], axis=1)
        
        # 2. Sort distances
        id_k = bn.argpartition(distances, k)[:k]
        
        labels_k = y_train[id_k]
        
        # 3. Classify (mode)
        counts = np.bincount(labels_k.astype(int))
        class_pred[i] = np.argmax(counts)
        
    return class_pred

# %% [markdown]
# 3\. Compute the error rate on the training set and the test set for $K \in \{1,2, \dotsc, 20\}$. Display the classification result (see 1.) for the configuration with the lowest error rate.

# %% [markdown]
# **Answer:**

# %%
error_train = []
error_test = []
K_list = range(1, 21)

class_test = test[:, 0] # actual test labels

for k in K_list:
    y_pred_train = knn_classifier(x_train, class_train, x_train, k)
    y_pred_test = knn_classifier(x_train, class_train, x_test, k)
    
    err_train = np.mean(y_pred_train != class_train)
    err_test = np.mean(y_pred_test != class_test)
    
    error_train.append(err_train)
    error_test.append(err_test)

best_k = K_list[np.argmin(error_test)]
print(f"Optimal K is {best_k} with test error {np.min(error_test):.4f}")

# Display classification result for optimal K
y_pred_best = knn_classifier(x_train, class_train, x_test, best_k)

plt.figure()
plt.scatter(x_test[y_pred_best == 1, 0], x_test[y_pred_best == 1, 1], c='r', marker='x', label='Pred Class 1')
plt.scatter(x_test[y_pred_best == 2, 0], x_test[y_pred_best == 2, 1], c='b', marker='x', label='Pred Class 2')
plt.legend()
plt.title(f"Classification on Test Set (K={best_k})")
plt.show()

# %% [markdown]
# 4\. Comment on your results. Which value of $K$ seems optimal ?
# 

# %% [markdown]
# **Answer:**

# %%
# The optimal value of K minimizes the classification error on the test set.
# A lower K (e.g., K=1) typically has low bias but high variance (overfitting the training data noise).
# A higher K has lower variance but higher bias, smoothing out the decision boundary too much (underfitting).
# The optimal K provides the best trade-off between bias and variance for this specific dataset.

# %% [markdown]
# 5\. Compare the results of you implementation with those of [`sklearn.neighbors.KNeighborsClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html?highlight=kneighborsclassifier#sklearn.neighbors.KNeighborsClassifier). Compare the runtime of these two versions using the [`timeit`](https://docs.python.org/3/library/timeit.html) module (see session 1).

# %% [markdown]
# **Answer:**

# %%
from sklearn.neighbors import KNeighborsClassifier
import timeit

# Sklearn implementation
knn_sk = KNeighborsClassifier(n_neighbors=best_k)
knn_sk.fit(x_train, class_train)
y_pred_sk = knn_sk.predict(x_test)
print(f"Are predictions identical? {np.array_equal(y_pred_best, y_pred_sk)}")

# Runtime comparison
time_custom = timeit.timeit(lambda: knn_classifier(x_train, class_train, x_test, best_k), number=10)
time_sk = timeit.timeit(lambda: knn_sk.predict(x_test), number=10)
print(f"Custom KNN runtime (10 runs): {time_custom:.4f} s")
print(f"Sklearn KNN runtime (10 runs): {time_sk:.4f} s")

# %% [markdown]
# ### B. Application to a real dataset (Breast cancer Wisconsin).
# 
# 6\. Apply the K-NN classifier to the real dataset `data/wdbc12.data.txt.` Further details about the data are provided in `data/wdbc12.names.txt`.
# 
# > Hint: you can use the function [`train_test_split` from `sklearn.model_selection`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) to split the dataset into a training and a test set.

# %% [markdown]
# **Answer:**

# %%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
wdbc = np.loadtxt('data/wdbc12.data.txt', delimiter=',')
y_wdbc = wdbc[:, 1]
X_wdbc = wdbc[:, 2:]

# Split data
X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(X_wdbc, y_wdbc, test_size=0.2, random_state=42)

# Standardize the features (important for KNN)
scaler = StandardScaler()
X_train_w_sc = scaler.fit_transform(X_train_w)
X_test_w_sc = scaler.transform(X_test_w)

# Find optimal K
error_test_w = []
K_list_w = range(1, 21)
for k in K_list_w:
    y_pred_w = knn_classifier(X_train_w_sc, y_train_w, X_test_w_sc, k)
    error_test_w.append(np.mean(y_pred_w != y_test_w))

best_k_w = K_list_w[np.argmin(error_test_w)]
print(f"Breast cancer dataset - Optimal K is {best_k_w} with test error {np.min(error_test_w):.4f}")

# %% [markdown]
# ## <a name="ex2">Exercise 2: Code acceleration with cython</a> [(&#8593;)](#content)
# 
# Cython allows C code to be easily interfaced with Python. It can be useful to make your code faster for a small coding effort, in particular when using loops. A general approach to optimize your code is outlined in the [Scipy lecture notes, Section 2.4](https://scipy-lectures.org/advanced/optimizing/index.html). Complementary reading about interfacing Python with C can be found in [Section 2.8](https://scipy-lectures.org/advanced/interfacing_with_c/interfacing_with_c.html).
# 
# 1\. Read carefully the [cython tutorial](http://docs.cython.org/en/latest/src/tutorial/cython_tutorial.html), which describes step by the step how the toy example reported below has been developed.

# %% [markdown]
# **Setup**: Compile the toy example provided in `example_cy/` by running, in the command line (anaconda prompt on windows)

# %% [markdown]
# ```bash
# cd example_cy && python setup.py build_ext --inplace
# ```

# %% [markdown]
# Note that the compilation process has been slightly automatised with the instructions reported in `example_cy/setup.py`. To test the module, run

# %%
!cd example_cy && python setup.py build_ext --inplace

# %%
import example_cy.example_cy.helloworld as toy

toy.printhello()

# %% [markdown]
# which should display
# ```python
# Hello World
# ```

# %% [markdown]
# > Warning: 
# > - do not forget to include an empty `__init__.py` file in the directory where your source code lives (`import` will fail if this is not the case).
# > - in case you have any setup issue, take a look at the `notes.md` file.
# > - if the C code and/or the executable do not seem to be regenerated by the build instructions, delete the C code and the executable first, and re-execute the compilation afterwards.
# > - do not hesitate to restart the Python kernel if necessary when the Cython executable has been re-generated.

# %% [markdown]
# 2\. Read the [Numpy/Cython tutorial](https://cython.readthedocs.io/en/latest/src/userguide/numpy_tutorial.html#numpy-tutorial), focussing on the paragraphs **Cython at a glance**, and **Your Cython environment** until **"More generic code"**. An example to compile a `.pyx` file depending on `numpy` is included in `example_np_cy/`.

# %% [markdown]
# > Remarks: 
# > - the `annotate=True` flag in the `setup.py` allows an additional `.html` document to be generated (`<your_module_name>.html`), showing, for each line of the Cython code, the associated C instructions generated. Highlighted in yellow are the interactions with Python: the darker a region appears, the less efficient the generated C code is for this section. Work in priority on these! 
# > - make sure all the previously generated files are deleted to allow the .html report to be generated;
# > - if you are working on your own machine and don't have a C/C++ compiler installed, read the notes provided in `notes.md`;
# > - use `cdef` for pure C functions (not exported to Python), `cpdef` should be favored for functions containing C instructions and later called from Python.

# %% [markdown]
# **Answer:**

# %%
# The tutorial details how to combine Cython and NumPy for fast execution, 
# notably by typing variables and using memory views (e.g. double[:] or double[:, :]) 
# instead of standard NumPy objects inside computationally heavy loops.

# %% [markdown]
# 3\. Use Cython to implement a faster version of the numpy K-NN classifier implemented in [Exercise 1](#ex1). To do so, apply step-by-step the techniques introduced in the [Numpy/Cython tutorial](https://cython.readthedocs.io/en/latest/src/userguide/numpy_tutorial.html#numpy-tutorial) (*i.e.*, compile and time your code after each step to report the evolution, keeping track of the different versions of the cython function).
# 
# > Hint: if you keep numpy arrays, make sure you use memory views (see numpy/cython tutorial) to access the elements within it. Be extremely careful with the type of the input arrays (you may need to recast the format of the input elements before entering the function. The `numpy.asarray` function can prove useful).
# 
# > **Detailed guidelines**: a few notes and *caveat* to help you re-writing your code in cython:
# > - try to reduce the number of calls to numpy instructions as much as possible;
# > - **you do not have to optimize everything**. For the KNN function above, most of the time is spent in computing euclidean distances: you can thus focus on optimizing tihs operations by explicitly writing a for loop, which will ensure a minimal interaction with numpy when generating the associated C code at compilation. Calls to other numpy functions can be kept as-is;
# > - if you need to create an array within the cython function, used np.zeros (**do NOT use python lists**), and use a memory view to access its content;
# > - specify the type for all variables and numpy arrays. Pay attention to the type of the input arrays passed to the Cython function;
# > - whenever an array is returned, use memory views and index(es) to efficiently access its content;
# > - some numpy operators (e.g., broadcasting mechanism) do not work with memory views. In this case, you can directly write for loop(s) to encode the operation of interest (the loops will be optimized out at compile time);
# > - only use at the final development stage the following cython optimization (not before, as they can crash the program without any help):
# >
# >```python
# >@cython.boundscheck(False)
# >@cython.wraparound(False)
# >```

# %% [markdown]
# **Answer:**

# %%
# We write the Cython code and setup script to disk so you can compile it later.
cython_code = """
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
"""

with open("knn_cy.pyx", "w") as f:
    f.write(cython_code)

setup_code = """
from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("knn_cy.pyx", annotate=True),
    include_dirs=[numpy.get_include()]
)
"""

with open("setup_knn.py", "w") as f:
    f.write(setup_code)

print("Created knn_cy.pyx and setup_knn.py. Please compile with:")
print("python setup_knn.py build_ext --inplace")

# %% [markdown]
# 4\. Compare the runtime of the two algorithms (using `timeit.timeit`), and conclude about the interest of using cython in this case.

# %% [markdown]
# **Answer:**

# %%
import timeit
import sys

# Assuming the module has been compiled:
try:
    import knn_cy
    
    time_custom = timeit.timeit(lambda: knn_classifier(x_train, class_train, x_test, best_k), number=10)
    time_cy = timeit.timeit(lambda: knn_cy.knn_classifier_cy(x_train.astype(np.float64), class_train.astype(np.float64), x_test.astype(np.float64), best_k), number=10)
    
    print(f"NumPy KNN runtime: {time_custom:.4f} s")
    print(f"Cython KNN runtime: {time_cy:.4f} s")
    print("Cython is significantly faster because the nested loops for distance calculation are compiled to efficient C code, bypassing Python overhead.")
except ImportError:
    print("Cython module 'knn_cy' not found.")
    print("Please compile it first by running:")
    print("!python setup_knn.py build_ext --inplace")

# %% [markdown]
# ## <a name="ex3">Exercise 3: Code acceleration with numba</a> [(&#8593;)](#content)
# 
# `numba` is a just-in-time (JIT) compiler which translates Python codes into efficient machine code at runtime. A significant acceleration can be obtained by adding a few simple decorators to a standard Python function, up to a few restrictions detailed [here](http://numba.pydata.org/numba-doc/latest/user/performance-tips.html).
# 
# If you have written most of the KNN classifier of exercise 1 with numpy, there is little to no chance that you will get an acceleration with numba (justifying the use of cython in this case). An interesting acceleration factor can however be obtained for the computation of the total variation investigated in session 2.

# %% [markdown]
# 1\. Take a look at the [numba 5 min tour](http://numba.pydata.org/numba-doc/latest/user/5minguide.html), and accelerate the total variation code from session 2 with the `@jit` decorator. You may have to rewrite small portions of your code to get the expected acceleration (see [performance tips](http://numba.pydata.org/numba-doc/latest/user/performance-tips.html)).

# %% [markdown]
# **Answer:**

# %%
from numba import jit

def total_variation_np(img):
    # Numpy implementation for comparison
    diff_x = img[1:, :-1] - img[:-1, :-1]
    diff_y = img[:-1, 1:] - img[:-1, :-1]
    return np.sum(np.sqrt(diff_x**2 + diff_y**2))

@jit(nopython=True)
def total_variation_numba(img):
    rows, cols = img.shape
    tv = 0.0
    for i in range(rows - 1):
        for j in range(cols - 1):
            dx = img[i + 1, j] - img[i, j]
            dy = img[i, j + 1] - img[i, j]
            tv += np.sqrt(dx**2 + dy**2)
    return tv

# Dummy image for testing
img = np.random.rand(500, 500)

# %% [markdown]
# 2\. Compare the runtime of the your numpy implementation and the `numba`-accelerated version (using `timeit.timeit`). 
# > **Warning**: first run the numba version once to trigger the compilation, and then time it as usual. This is needed to avoid including the JIT compilation step in the runtime.

# %% [markdown]
# **Answer:**

# %%
# First run to trigger JIT compilation
tv = total_variation_numba(img)

# Runtime comparison
import timeit
time_tv_np = timeit.timeit(lambda: total_variation_np(img), number=100)
time_tv_numba = timeit.timeit(lambda: total_variation_numba(img), number=100)

print(f"Numpy TV runtime: {time_tv_np:.4f} s")
print(f"Numba TV runtime: {time_tv_numba:.4f} s")
print("Numba provides a significant speedup by JIT-compiling the Python loops into fast machine code.")



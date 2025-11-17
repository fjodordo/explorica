import pandas as pd
import numpy as np

DATA_SEQUENCES = [
    # === 1D structures ===
    [1, 2, 3, 4, 100],                        # Python list
    (1, 2, 3, 4, 100),                        # Tuple
    np.array([1, 2, 3, 4, 100]),              # NumPy 1D array
    pd.Series([1, 2, 3, 4, 100]),             # pandas Series

    # === 1D edge variants ===
    range(1, 6),                              # range object
    (x for x in [1, 2, 3, 4, 100]),           # generator
    np.array([1., 2., 3., 4., 100.], dtype=np.float64),  # explicit dtype
    pd.Index([1, 2, 3, 4, 100]),              # pandas Index

    # === 2D structures ===
    [[1, 2, 3], [4, 5, 100]],                 # list of lists
    tuple(zip([1, 4], [2, 5], [3, 100])),     # tuple of tuples (transposed)
    np.array([[1, 2, 3], [4, 5, 100]]),       # 2D numpy array
    pd.DataFrame({
        "A": [1, 2, 3, 4, 100],
        "B": [2, 3, 4, 5, 200]
    }),                                       # pandas DataFrame

    # === 2D exotic types ===
    # np.matrix([[1, 2, 3], [4, 5, 100]]),      # legacy numpy matrix
    pd.DataFrame(np.array([[1, 2], [3, 100]]), columns=["x", "y"]),  # DataFrame from ndarray
    pd.DataFrame.from_records(
        [(1, 2, 3), (4, 5, 100)], columns=["a", "b", "c"]
    ),                                        # DataFrame from records
]

DETECT_METHODS = ["iqr", "zscore"]

DF_WITH_NAN = pd.DataFrame({"A": [0, 1, 2, 3],
                       "B": [0, 1, 2, None]})

DESCRIBE_DISTRIBUTIONS_STRUCTURE = ["skewness", "kurtosis", "is_normal", "desc"]
from functools import partial
import multiprocessing as mp
import logging as log

from numba import jit, prange
from math import sqrt
import numpy as np

@jit(nopython=True, parallel=True)
def _gasf(X_cos, X_sin, n_samples, image_size):
    X_gasf = np.empty((n_samples, image_size, image_size))
    for i in prange(n_samples):
        X_gasf[i] = np.outer(X_cos[i], X_cos[i]) - np.outer(X_sin[i], X_sin[i])
    return X_gasf


@jit(nopython=True, parallel=True)
def _gadf(X_cos, X_sin, n_samples, image_size):
    X_gadf = np.empty((n_samples, image_size, image_size))
    for i in prange(n_samples):
        X_gadf[i] = np.outer(X_sin[i], X_cos[i]) - np.outer(X_cos[i], X_sin[i])
    return X_gadf

@jit(nopython=True, parallel=True)
def compute_GM_optim(STS: np.ndarray, 
                    patterns: np.ndarray, 
                    method: str = 'summation',
                    sample_range: tuple[int] = (-1, 1)
                    ) -> np.ndarray:

    """ Computes the gramian matrix (GM) for a given set of patterns and a given STS.
        Optimized version using Numba.
        
        The GM has dimensions (n_patts, l_patts, STS_length), where n_patts is the number of patterns,
        l_patts is the length of the patterns, and STS_length is the length of the STS.
        
        Parameters
        ----------
        STS : np.ndarray
            The STS to compute the DM for.
        patterns : np.ndarray
            The patterns used to compute the DM.

        References
    ----------
    .. [1] Z. Wang and T. Oates, "Encoding Time Series as Images for Visual
           Inspection and Classification Using Tiled Convolutional Neural
           Networks." AAAI Workshop (2015).
    """

    """
    X = check_array(X)
        n_samples, n_timestamps = X.shape
        image_size = self._check_params(n_timestamps)

        paa = PiecewiseAggregateApproximation(
            window_size=None, output_size=image_size,
            overlapping=self.overlapping
        )
        X_paa = paa.fit_transform(X)
        if self.sample_range is None:
            X_min, X_max = np.min(X_paa), np.max(X_paa)
            if (X_min < -1) or (X_max > 1):
                raise ValueError("If 'sample_range' is None, all the values "
                                 "of X must be between -1 and 1.")
            X_cos = X_paa
        else:
            scaler = MinMaxScaler(sample_range=self.sample_range)
            X_cos = scaler.fit_transform(X_paa)
        X_sin = np.sqrt(np.clip(1 - X_cos ** 2, 0, 1))
        if self.method in ['s', 'summation']:
            X_new = _gasf(X_cos, X_sin, n_samples, image_size)
        else:
            X_new = _gadf(X_cos, X_sin, n_samples, image_size)

        if self.flatten:
            return X_new.reshape(n_samples, -1)
        return X_new
    """

    n_patts: int = patterns.shape[0]
    l_patts: int = patterns.shape[1]
    l_STS: int = STS.shape[0]

    X = STS

    if self.sample_range is None:
            X_min, X_max = np.min(X_paa), np.max(X_paa)
            if (X_min < -1) or (X_max > 1):
                raise ValueError("If 'sample_range' is None, all the values "
                                 "of X must be between -1 and 1.")
            X_cos = X_paa
        else:
            scaler = MinMaxScaler(sample_range=sample_range)
            X_cos = scaler.fit_transform(X_paa)
    X_sin = np.sqrt(np.clip(1 - X_cos ** 2, 0, 1))


     


    # Compute the Gramian distance matrix
    GM = np.empty((n_patts, l_patts, l_STS), dtype=np.float32)
    
    if method in ['s', 'summation']:
        for p in prange(n_patts):
            for i in prange(l_STS):
                for j in prange(l_patts):
                    GM[p, j, i] = STS_cos[i]*patterns_cos[p, j] + STS_sin[i]*patterns_sin[p, j]
    elif method in ["d, difference"]:
         for p in prange(n_patts):
            for i in prange(l_patts):
                for j in prange(l_STS):
                    GM[p, j, i] = STS_cos[i]*patterns_cos[p, j] + STS_sin[i]*patterns_sin[p, j]

    # Return the full distance matrix
    return GM


if __name__ == "__main__":


    """ Small test script for optimization. """

    import time
    import matplotlib.pyplot as plt 

    STS = np.sin(np.linspace(0, 6*np.pi, 10000))
    lpat = 300
    patterns = np.stack([np.arange(0, lpat), np.zeros(lpat), np.arange(0, lpat)[::-1]])
    # standardize patterns
    patterns = (patterns - np.mean(patterns, axis=1, keepdims=True)) / np.std(patterns, axis=1, keepdims=True)
    print(patterns)

    start = time.perf_counter()
    DM1 = compute_DM(STS, patterns, 0.1)
    end = time.perf_counter()
    print("Elapsed (baseline) = {}s".format((end - start)))

    start = time.perf_counter()
    DM2 = compute_GM_optim(STS, patterns, 0.1)
    end = time.perf_counter()
    print("Elapsed (with compilation) = {}s".format((end - start)))

    start = time.perf_counter()
    DM2 = compute_GM_optim(STS, patterns, 0.1)
    end = time.perf_counter()
    print("Elapsed (with compilation) = {}s".format((end - start)))

    
    # plt.plot(STS)
    # plt.plot(patterns[0])
    # plt.plot(patterns[1])
    
    # plt.figure()
    # plt.imshow(DM1[0])

    # plt.figure()
    # plt.imshow(DM2[0])

    
    plt.show()



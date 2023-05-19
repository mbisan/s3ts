from functools import partial
import multiprocessing as mp
import logging as log

from numba import jit, prange
from math import sqrt
import numpy as np

@jit(nopython=True, parallel=True)
def compute_GM_optim(STS: np.ndarray, patterns: np.ndarray, rho: float):

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
        rho : float
            The memory parameter.
    """

    n_patts: int = patterns.shape[0]
    l_patts: int = patterns.shape[1]
    l_STS: int = STS.shape[0]
    
    w: float = rho ** (1 / l_patts)

    # Compute point-wise distance
    DM = np.zeros((n_patts, l_patts, l_STS), dtype=np.float32)
    for p in prange(n_patts):
        for i in prange(l_patts):
            for j in prange(l_STS):
                DM[p, i, j] = sqrt((patterns[p,i] - STS[j])**2)

    # Compute the distance matrix
    for p in prange(n_patts):

        # Solve first row
        for j in range(1, l_STS):
            DM[p,0,j] += w*DM[p,0,j-1]

        # Solve first column
        for i in range(1, l_patts):
            DM[p,i,0] += DM[p,i-1,0]

        # Solve the rest
        for i in range(1, l_patts):
            for j in range(1, l_STS):
                DM[p,i,j] += min([DM[p,i-1,j], w*DM[p,i-1,j-1], w*DM[p,i,j-1]])

    # Return the full distance matrix
    return DM


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



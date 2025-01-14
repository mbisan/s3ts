from functools import partial
import multiprocessing as mp
import logging as log

from numba import jit, prange
from math import sqrt
import numpy as np

class OESM:
    
    """
    On-line Elastic Similarity class

    Parameters
    ----------
    R : 1-D array_like
        time series pattern
    w : float
        On-line similarity measure memory. Must be between 0 and 1.
    rateX : float, optional, default: 1
        Reference Time series generate_files rate
    rateY : float, optional, default: None
        Query Time series generate_files rate. If rateY = None it takes rateX value
    dist : string, optional, default: 'euclidean'
        cost function
        OPTIONS:
        'euclidean': np.sqrt((x_i - y_j)**2)
        'edr':       1 if abs(x_i - y_j) >= epsilon else 0
        'edit':      1 if x_i != y_j else 0
        'erp':       abs(x_i - y_j) if abs(x_i - y_j) >= epsilon else 0
    epsilon : float, optional, default: None
        edr threshold parameter
    """

    def __init__(self, 
            R: np.ndarray, 
            w: float, 
            dist: str = 'euclidean', 
            epsilon: float = None
            ) -> None:

        # Check if distance metric choice is valid
        valid_dist = ['euclidean', 'edr', 'erp', 'edit']
        if dist not in valid_dist:
            raise ValueError(f"dist must be one of {valid_dist}")
        self.dist = dist

        if isinstance(R, (np.ndarray)) and R.size > 2:
            self.R = R

        if (w >= 0) and (w <= 1):
            self.w = w
        else:
            raise ValueError('w must be between 0 and 1')

        if (epsilon is None) and (dist in ['edr', 'erp']):
            raise ValueError(
                'for dist edr or erp epsilon must be a non negative float')
        elif (epsilon is not None) and epsilon < 0:
            raise ValueError('epsilon must be a non negative float')
        self.epsilon = epsilon

    def init_dist(self, S: np.ndarray):

        """
        Initial Similarity Measure

        Parameters
        ----------
        S : 1-D array_like
            Array containing time series observations

        Returns
        -------
        dist: float
            Elastic Similarity Measure between R (pattern time series) and S (query time series)
        """

        if isinstance(S, (np.ndarray)) and S.size > 2:
            pass
        else:
            raise ValueError('S time series must have more than 2 instances')

        # Compute point-wise distance
        if self.dist == 'euclidean':
            RS = self.__euclidean(self.R, S)
        elif self.dist == 'edr':
            RS = self.__edr(self.R, S, self.epsilon)
        elif self.dist == 'erp':
            RS = self.__erp(self.R, S, self.epsilon)
        elif self.dist == 'edit':
            RS = self.__edit(self.R, S)

        # compute recursive distance matrix using dynamic programing
        r, s = np.shape(RS)

        # Solve first row
        for j in range(1, s):
            RS[0, j] += self.w * RS[0, j - 1]

        # Solve first column
        for i in range(1, r):
            RS[i, 0] += RS[i - 1, 0]

        # Solve the rest
        for i in range(1, r):
            for j in range(1, s):
                RS[i, j] += np.min([RS[i - 1, j], self.w *
                                    RS[i - 1, j - 1], self.w * RS[i, j - 1]])

        # save statistics
        self.dtwR = RS[:, -1]

        return RS

    def update_dist(self, Y: np.ndarray):

        '''
        Add new observations to query time series

        Parameters
        ----------
        Y : array_like or float
            new query time series observation(s)

        Returns
        -------
        dist : float
            Updated distance

        Warning: the time series have to be non-empty
        (at least composed by a single measure)

          -----------
        R | RS | RY |
          -----------
            S    Y
        '''

        if isinstance(Y, (np.ndarray)):
            pass
        else:
            Y = np.array([Y])

        # Solve RY
        dtwRY = self._solveRY(Y, self.dtwR)

        # Save statistics
        self.dtwR = dtwRY

        return dtwRY

    def __euclidean(self, X, Y):
        Y_tmp, X_tmp = np.meshgrid(Y, X)
        XY = np.sqrt((X_tmp - Y_tmp)**2)
        return XY

    def __edr(self, X, Y, epsilon):
        Y_tmp, X_tmp = np.meshgrid(Y, X)
        XY = 1.0 * (abs(X_tmp - Y_tmp) < epsilon)
        return XY

    def __erp(self, X, Y, epsilon):
        Y_tmp, X_tmp = np.meshgrid(Y, X)
        XY = abs(X_tmp - Y_tmp)
        XY[XY < epsilon] = 0
        return XY

    def __edit(self, X, Y):
        Y_tmp, X_tmp = np.meshgrid(Y, X)
        XY = 1.0 * (X_tmp == Y_tmp)
        return XY

    def _solveRY(self, Y, dtwR):

        '''
        R, Y: to (partial) time series
         -----------
        R|prev| RY |
         -----------
           S    Y

        dtwR: partial solutions of DTW(R,S)
        iniI: Index of the first point of R in the complete time series
        iniJ: Index of the first point of Y in the complete time series

        * Warning *: R and Y have to be non-empty (partial) series
        '''

        # Compute point-wise distance
        if self.dist == 'euclidean':
            RY = self.__euclidean(self.R, Y)
        elif self.dist == 'edr':
            RY = self.__edr(self.R, Y, self.epsilon)
        elif self.dist == 'erp':
            RY = self.__erp(self.R, Y, self.epsilon)
        elif self.dist == 'edit':
            RY = self.__edit(self.R, Y)

        r, n = np.shape(RY)

        # First first column
        RY[0, 0] += self.w * dtwR[0]
        for i in range(1, r):
            RY[i, 0] += np.min([
                RY[i - 1, 0], self.w * dtwR[i - 1], self.w * dtwR[i]])

        # Solve first row
        for j in range(1, n):
            RY[0, j] += self.w * RY[0, j - 1]

        # Solve the rest
        for j in range(1, n):
            for i in range(1, r):
                RY[i, j] += np.min([RY[i - 1, j], self.w *
                                    RY[i - 1, j - 1], self.w * RY[i, j - 1]])

        return RY[:, -1]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def compute_patt_DM(patt_idx: np.ndarray, rho: float,
        STS: np.ndarray, patterns: np.ndarray, 
        dist: str = "euclidean"):
    
    """ Wrapper that calculates the distance matrix for given single pattern. """

    STS_length: int = STS.shape[0]
    l_patts: int = patterns.shape[1]

    init_width = 3
    oesm = OESM(R = patterns[patt_idx], w=rho, dist=dist)
    patt_DM = np.empty((l_patts, STS_length), dtype=np.float64)
    patt_DM[:,:init_width] = oesm.init_dist(STS[:init_width])
    for point in range(init_width, len(STS)):
        patt_DM[:, point] = oesm.update_dist(STS[point:point+1])

    return patt_DM

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def compute_DM(STS: np.ndarray, patterns: np.ndarray, rho: float, 
        dist: str = "euclidean", num_workers: int = mp.cpu_count()):

    """ Computes the dissimilarity matrix (DM) for a given set of patterns and a given STS. 
    THe DM has dimensions (n_patts, l_patts, STS_length), where n_patts is the number of patterns,
    l_patts is the length of the patterns, and STS_length is the length of the STS.

    Parameters
    ----------
    STS : np.ndarray
        The STS to compute the DM for.
    patterns : np.ndarray
        The patterns used to compute the DM.
    rho : float
        The memory parameter.
    nprocs : int, optional
        The number of processes to use, by default mp.cpu_count()
    """

    # Get the number of patterns
    n_patts = patterns.shape[0]

    # Get the length of the patterns
    l_samp = patterns.shape[1]

    # Scale memory parameters
    scaled_rho = rho ** (1 / l_samp)

    # List of pattern IDs
    patt_ids = np.arange(n_patts).tolist()

    # Function call to compute the distance matrix for each pattern ID
    compute_patt_DM_call = partial(compute_patt_DM, STS=STS, patterns=patterns, rho=scaled_rho, dist=dist)

    # Compute the distance matrix for each pattern ID in parallel
    with mp.Pool(processes=min(mp.cpu_count(), n_patts)) as pool:
        full_DM = pool.map(compute_patt_DM_call, patt_ids)

    # Return the full distance matrix
    return np.array(full_DM)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

@jit(nopython=True, parallel=True)
def compute_DM_optim(STS: np.ndarray, patterns: np.ndarray, rho: float):

    """ Computes the dissimilarity matrix (DM) for a given set of patterns and a given STS.
        Optimized version using Numba.
        
        The DM has dimensions (n_patts, l_patts, STS_length), where n_patts is the number of patterns,
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
    DM2 = compute_DM_optim(STS, patterns, 0.1)
    end = time.perf_counter()
    print("Elapsed (with compilation) = {}s".format((end - start)))

    start = time.perf_counter()
    DM2 = compute_DM_optim(STS, patterns, 0.1)
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



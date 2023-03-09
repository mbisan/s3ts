# sktime
from sktime.clustering.k_medoids import TimeSeriesKMedoids

# numpy / scipy
from scipy.spatial import distance_matrix
from math import ceil
import logging as log
import numpy as np

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def compute_medoids(
        X: np.ndarray, Y: np.ndarray,
        distance_type: str = 'euclidean'
    ) -> tuple[np.ndarray, np.ndarray]: 

    """ Computes the medoids of the classes in the dataset. """

    assert(len(X.shape) == 2)
    s_length = X.shape[1]

    nclasses = len(np.unique(Y))

    # Find the medoids for each class
    medoids = np.empty((nclasses, s_length), dtype=float)
    medoid_ids = np.empty(nclasses, dtype=int)
    for i, y in enumerate(np.unique(Y)):

        index = np.argwhere(Y == y)
        Xi = X[index, :]

        # ...using simple euclidean distance        
        if distance_type == "euclidean":
            medoid_idx = np.argmin(distance_matrix(Xi.squeeze(), Xi.squeeze()).sum(axis=0))
            medoids[i,:] = Xi[medoid_idx,:]
            medoid_ids[i] = index[medoid_idx]

        # ...using Dynamic Time Warping (DTW)
        elif distance_type == "dtw":
            tskm = TimeSeriesKMedoids(n_clusters=1, init_algorithm="forgy", metric="dtw")
            tskm.fit(Xi)
            medoids[i,:] = tskm.cluster_centers_.squeeze()
            medoid_ids[i] = np.where(np.all(Xi.squeeze() == medoids[i,:], axis=1))[0][0]

        else:
            raise NotImplementedError

    return medoids, medoid_ids

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def compute_STS(
        X: np.ndarray, 
        Y: np.ndarray,
        target_nframes: int,
        frame_buffer: int = 0,
        random_state: int = 0,
        ) -> tuple[np.ndarray, np.ndarray]:

    """
    Builds an STS from and array of samples and labels.
    """

    assert(len(X.shape) == 2)
    assert(X.shape[0] == Y.shape[0])
    rng = np.random.default_rng(seed=random_state)
    
    nsamples = X.shape[0]
    l_sample = X.shape[1]
    
    # recommended number of frames
    rec_nframes = nsamples*l_sample

    randomized = True
    if target_nframes is None:
        target_nframes = rec_nframes
        randomized = False
    else: 
        target_nframes: float = target_nframes*rec_nframes

    log.info(f"Sample size: {target_nframes}")
    log.info(f"Number of samples: {nsamples}")
    log.info(f"Target number of frames: {target_nframes}")
    log.info(f"Recom. number of frames: {rec_nframes}")

    if target_nframes < rec_nframes:
        log.warn(f"Target number of frames {target_nframes} below " +\
              f"recommended {rec_nframes} for {nsamples} samples of size {l_sample}")

    target_nsamples = ceil((target_nframes + frame_buffer)/float(l_sample))

    STS_X = np.empty(target_nsamples*l_sample)
    STS_Y = np.empty(target_nsamples*l_sample)

    if randomized:
        for r in range(target_nsamples):
            random_idx = rng.integers(0, nsamples)
            STS_X[r*l_sample:(r+1)*l_sample] = X[random_idx,:]
            STS_Y[r*l_sample:(r+1)*l_sample] = Y[random_idx]
    else:
        buf = target_nsamples-nsamples
        for r in range(buf):
            random_idx = rng.integers(0, nsamples)
            STS_X[r*l_sample:(r+1)*l_sample] = X[random_idx,:]
            STS_Y[r*l_sample:(r+1)*l_sample] = Y[random_idx]
        
        for s, random_idx in enumerate(rng.permutation(np.arange(nsamples))):
            STS_X[(buf+s)*l_sample:(buf+s+1)*l_sample] = X[random_idx,:]
            STS_Y[(buf+s)*l_sample:(buf+s+1)*l_sample] = Y[random_idx]
        
    return STS_X, STS_Y, target_nframes
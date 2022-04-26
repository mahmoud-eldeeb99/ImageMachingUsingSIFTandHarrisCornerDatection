import numpy as np
from scipy.spatial.distance import cdist
import time


def match(descriptors1, descriptors2, option, distance_ratio,max_distance=np.inf):
    
    if(option == 'ssd'):
        ssd_time_start = time.time()
        distances = cdist(descriptors1, descriptors2, metric = 'sqeuclidean')   # distances.shape: [len(d1), len(d2)]
    
    if(option == 'corr'):
        corr_time_start = time.time()
        distances = cdist(descriptors1, descriptors2, metric = 'correlation')
        
        
    indices1 = np.arange(descriptors1.shape[0])     # [0, 1, 2, 3, 4, 5, 6, 7, ..., len(d1)] "indices of d1"
    indices2 = np.argmin(distances, axis=1)         # [12, 465, 23, 111, 123, 45, 67, 2, 265, ..., len(d1)] "list of the indices of d2 points that are closest to d1 points"
                                                    # Each d1 point has a d2 point that is the most close to it.

    matches1 = np.argmin(distances, axis=0)     # [15, 37, 283, ..., len(d2)] "list of d1 points closest to d2 points"
                                                # Each d2 point has a d1 point that is closest to it.
    # indices2 is the forward matches [d1 -> d2], while matches1 is the backward matches [d2 -> d1].
    mask = indices1 == matches1[indices2]       # len(mask) = len(d1)
    # we are basically asking does this point in d1 matches with a point in d2 that is also matching to the same point in d1 ?
    indices1 = indices1[mask]
    indices2 = indices2[mask]
    
    if max_distance < np.inf:
        mask = distances[indices1, indices2] < max_distance
        indices1 = indices1[mask]
        indices2 = indices2[mask]


    modified_dist = distances
    fc = np.min(modified_dist[indices1,:], axis=1)
    modified_dist[indices1, indices2] = np.inf
    fs = np.min(modified_dist[indices1,:], axis=1)
    mask = fc/fs <= distance_ratio
    indices1 = indices1[mask]
    indices2 = indices2[mask]

    # sort matches using distances
    dist = distances[indices1, indices2]
    sorted_indices = dist.argsort()

    matches = np.column_stack((indices1[sorted_indices], indices2[sorted_indices]))
    if(option == 'ssd'):
        ssd_time_end = time.time()
        print(f"Execution time of SSD matching is {ssd_time_end - ssd_time_start}  sec")
    
    if(option == 'corr'):
        corr_time_end = time.time()
        print(f"Execution time of CORR matching is {corr_time_end - corr_time_start}  sec")
    
    
    return matches



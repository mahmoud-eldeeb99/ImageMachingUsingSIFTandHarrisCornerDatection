import numpy as np
from scipy import signal
import time



# key_point orientation assignment
def assign_orientation(kps, img, num_bins=36):
  feature_desc_time_start = time.time()
  SIGMA = 1.6   # according to  the paper
  new_kps = []
  sigma = SIGMA * 1.5
  w = int(2 * np.ceil(sigma) + 1)
  kernel = gaussian_filter((w, w), sigma)
  bin_width = 360 // num_bins

  for kp in kps:
    x, y = int(kp[0]), int(kp[1])
    hist = np.zeros(num_bins, dtype=np.float32)
    # 7x7 window around the key point
    sub_img = np.array(img[x - 3:x + 4, y - 3:y + 4])

    mag, Dir = sobel_edge(sub_img)

    # weighted_mag = np.multiply(mag, kernel)
    weighted_mag = np.dot(mag, kernel)

    # calculating the  histogram
    for i in range(0, len(Dir)):
      for j in range(0, len(mag)):
        bin_indx = int(np.floor(Dir[i][j]) // bin_width)
        hist[bin_indx] += weighted_mag[i][j]

    # finding new key points (value > 0.8 * max_val)
    max_val = np.max(hist)
    for bin_no, val in enumerate(hist):
      if .8 * max_val <= val:
        angle = (bin_no + 0.5) * (360. / num_bins) % 360
        new_kps.append([kp[0], kp[1], angle])

  return np.array(new_kps), feature_desc_time_start



# Descriptor Generation
def feature_descriptor(kepy_points, img, time_start, num_subregion=4, num_bin=8):
    descriptors = []
    sigma = 1.5
    kernel = gaussian_filter((16, 16), sigma)
    for kp in kepy_points:
        x, y, dominant_angle = int(kp[0]), int(kp[1]), kp[2]

        # 16 X 16 window around the key point
        sub_img = img [x - 8:x + 8, y - 8:y + 8]
        mag, dir = sobel_edge(sub_img)
        weighted_mag = np.dot(mag, kernel)
        # subtract the dominant direction
        dir = (((dir - dominant_angle) % 360) * num_bin / 360.).astype(int)
        features = []
        for sub_i in range(num_subregion):
            for sub_j in range(num_subregion):
                sub_weights = weighted_mag[sub_i * 4:(sub_i + 1) * 4, sub_j * 4:(sub_j + 1) * 4]
                sub_dir_idx = dir[sub_i * 4:(sub_i + 1) * 4, sub_j * 4:(sub_j + 1) * 4]
                hist = np.zeros(num_bin, dtype=np.float32)
                for bin_idx in range(num_bin):
                    hist[bin_idx] = np.sum(sub_weights[sub_dir_idx == bin_idx])
                features.extend(hist.tolist())
        features /= (np.linalg.norm(np.array(features)))
        features = np.clip(features, np.finfo(np.float16).eps, 0.2)
        features /= (np.linalg.norm(features))
        descriptors.append(features)
    feature_desc_time_end = time.time()
    print(f"Execution time of the feature descriptor generation is {feature_desc_time_end - time_start}  sec")
    return descriptors


# helper
def gaussian_filter(shape, sigma):
  m, n = [(ss - 1.) / 2. for ss in shape]
  y, x = np.ogrid[-m:m + 1, -n:n + 1]
  kernel = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
  kernel[kernel < np.finfo(kernel.dtype).eps * kernel.max()] = 0
  sum_kernel = kernel.sum()
  if sum_kernel != 0:
    kernel /= sum_kernel

  return kernel

def sobel_edge(img):
  kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
  ky = (kx.transpose())
  Ix = signal.convolve2d(img, kx, boundary='symm', mode='same')
  Iy = signal.convolve2d(img, ky, boundary='symm', mode='same')
  Magnitude = np.sqrt(Ix * Ix + Iy * Iy)
  direction = np.rad2deg( np.arctan2(Iy, Ix)) % 360
  return Magnitude, direction



# Test


    
  


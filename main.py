import cv2
import Harris_Corner_Detector as HC
import SIFT_Descriptors as SIFT
import numpy as np
import MachingFuncatin
from skimage.feature import plot_matches
import matplotlib.pylab as plt




image1 = cv2.imread("Original Image.jpg", cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread("Rotated Image.jpg", cv2.IMREAD_GRAYSCALE)


image1 = cv2.resize(image1, (500, 500), interpolation=cv2.INTER_AREA)
image2 = cv2.resize(image2, (500, 500), interpolation=cv2.INTER_AREA)
# Display the original image
# cv2.imwrite("result_gray.jpg", image1)


# Display the query image
# cv2.imwrite("result1_gray.jpg", image2)


# apply functions using Image 1 ,2
R = HC.HarrisCornerDetection(image1)
kps, time_start = SIFT.assign_orientation(R, image1)
des = SIFT.feature_descriptor(kps, image1, time_start, num_subregion=4, num_bin=8)
# print("Descriptor1:",des)
R2 = HC.HarrisCornerDetection(image2)
kps2, time_start2 = SIFT.assign_orientation(R2, image2)
des2 = SIFT.feature_descriptor(kps2, image2, time_start2, num_subregion=4, num_bin=8)
# print("Descriptor:",des2)
des = np.array(des)
des2 = np.array(des2)

matches = MachingFuncatin.match(des, des2, 'ssd', distance_ratio=0.5, max_distance=np.inf)

# plot using matplot and then save it as picture
fig = plt.figure(figsize=(20, 10))

ax = fig.add_subplot(1, 1, 1)

plot_matches(ax, image1, image2, kps, kps2, matches[:200],
             alignment='horizontal', only_matches=True)

fig.savefig("FinalMatchingImage.jpg")
fig.clf()


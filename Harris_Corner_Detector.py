from scipy import signal
from numpy.linalg import det
import numpy as np
import time
import cv2
import matplotlib.pylab as plt

def HarrisCornerDetection(image):
    #set start time
    Harris_time_start = time.time()
    Width, Hight = image.shape
    # edge detection mag
    Xgradient, Ygradient = sobelEdgeDetection(image)



    # # Eliminate the negative
    for i in range(Width):
        for j in range(Hight):
            if Ygradient[i][j] < 0:
                Ygradient[i][j] *= -1
            if Xgradient[i][j] < 0:
                Xgradient[i][j] *= -1

   # Matrix M calculations
    Xgradient2 = np.square(Xgradient)
    Ygradient2 = np.square(Ygradient)
    ImgXY = np.multiply(Xgradient, Ygradient)
    ImgYX = np.multiply(Ygradient, Xgradient)

    #Use Gaussian Blur
    Sigma = 1.4
    kernelsize = (3, 3)

    Xgradient2 = GaussianFilter(Xgradient2, kernelsize, Sigma)
    Ygradient2 = GaussianFilter(Ygradient2, kernelsize, Sigma)
    ImgXY = GaussianFilter(ImgXY, kernelsize, Sigma)
    ImgYX = GaussianFilter(ImgYX, kernelsize, Sigma)

    alpha = 0.06
    R = np.zeros((Width, Hight), np.float32)
    # For every pixel find the corner strength
    for row in range(Width):
        for col in range(Hight):
            M_bar = np.array([[Xgradient2[row][col], ImgXY[row][col]], [ImgYX[row][col], Ygradient2[row][col]]])
            R[row][col] = np.linalg.det(M_bar) - (alpha * np.square(np.trace(M_bar)))

    # Empirical Parameter
    # This parameter will need tuning based on the use-case
    CornerStrengthThreshold = 600000

    Key_Points = []
    # Look for Corner strengths above the threshold
    for row in range(Width):
        for col in range(Hight):
            if R[row][col] > CornerStrengthThreshold:
                # print(R[row][col])
                max = R[row][col]

                # Local non-maxima suppression
                skip = False
                for nrow in range(5):
                    for ncol in range(5):
                        if row + nrow - 2 < Width and col + ncol - 2 < Hight:
                            if R[row + nrow - 2][col + ncol - 2] > max:
                                skip = True
                                break

                if not skip:
                    # Point is expressed in x, y which is col, row
                    # cv2.circle(bgr, (col, row), 1, (0, 0, 255), 1)
                    Key_Points.append((row, col))
    Harris_time_end = time.time()
    print(f"Execution time of the Harris corner Detector is {Harris_time_end - Harris_time_start}  sec")
    return Key_Points

# helper
def GaussianFilter(img, shape, sigma):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh

    return signal.convolve2d(img, h)

def sobelEdgeDetection(img):
    row, col = img.shape
    Ix = np.zeros([row, col])
    Iy = np.zeros([row, col])

    kx = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]])
    ky = (kx.transpose())
    for i in range(1, row - 2):
        for j in range(1, col - 2):
            Ix[i][j] = np.sum(np.multiply(kx, img[i:i + 3, j:j + 3]))
            Iy[i][j] = np.sum(np.multiply(ky, img[i:i + 3, j:j + 3]))

    return Ix, Iy


#### >> test ####
#
# # Get the first image
# firstimage = cv2.imread('corner.jpg', cv2.IMREAD_GRAYSCALE)
# w, h = firstimage.shape
# # Covert image to color to draw colored circles on it
# bgr = cv2.cvtColor(firstimage, cv2.COLOR_GRAY2RGB)
# # Corner detection
# R = HarrisCornerDetection(firstimage)
#
# # # Display image indicating corners and save it
# # cv2.imshow("Corners",bgr)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
#
#
#
#
# # test for corner detection
# implot = plt.imshow(bgr)
# for p,q in R:
#     x_cord = p # try this change (p and q are already the coordinates)
#     y_cord = q
#     plt.scatter([x_cord], [y_cord])
# plt.show()

# Hough Transform Demo
# Finding the most significant straight contours in a given image

import cv2
import numpy as np
import math

def compute_edge_image(bgr_img):
    """ Compute the edge magnitude of an image using a pair of Sobel filters """

    sobel_v = np.array([[-1, -2, -1],   # Sobel filter for the vertical gradient. Note that the filter2D function computes a correlation
                        [ 0,  0, 0],    # instead of a convolution, so the filter is *not* rotated by 180 degrees.
                        [ 1,  2, 1]])
    sobel_h = sobel_v.T                 # The convolution filter for the horizontal gradient is simply the transpose of the previous one

    gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    gradient_v = cv2.filter2D(gray_img, ddepth=cv2.CV_32F, kernel=sobel_v)
    gradient_h = cv2.filter2D(gray_img, ddepth=cv2.CV_32F, kernel=sobel_h)
    gradient_magni = np.sqrt(gradient_v**2 + gradient_h**2)
    near_max = np.percentile(gradient_magni, 99.5)      # Clip magnitude at percentile 99.5 to prevent outliers from determining the range of relevant magnitudes
    edge_img = np.clip(gradient_magni * 255.0 / near_max, 0.0, 255.0).astype(np.uint8)  # Normalize and convert magnitudes into grayscale image
    return edge_img

def hough_transform_circles(input_space, num_row_bins, num_col_bins, num_radius_bins, min_radius, max_radius):
    """" Perform Hough transform of an image to an alpha-d space represemnting circles """

    output_space = np.zeros((num_row_bins, num_col_bins, num_radius_bins), dtype=int)

    edge_pixels = np.row_stack(np.nonzero(input_space >= 150))   # Only consider edges exceeding a threshold; higher threshold speeds up computation but ignores more edges

    # Loop over all edge pixels
    edge_pixels = zip(edge_pixels[0], edge_pixels[1])
    for i, j in edge_pixels:
        # Loop over all pixels within rmax of (i, j)
        for v in range(max(0, i - max_radius), min(num_row_bins, i + max_radius + 1)):
            for h in range(max(0, j - max_radius), min(num_col_bins, j + max_radius + 1)):

                # Compute distance between (i, j) and (v, h)
                dist = np.sqrt((i - v)**2 + (j - h)**2)

                # Determine radius bin
                r = int(dist)
                if r >= max_radius:
                    continue

                # Add a vote to accumulator array
                output_space[v, h, r] += 1


    return output_space

def find_maxima(hough_space, num_maxima=1, threshold=None):
    """
    Finds the local maxima in the Hough transform circle output space.

    Parameters
    ----------
    hough_space : numpy.ndarray
        The Hough transform circle output space.
    num_maxima : int, optional
        The number of maxima to return. Defaults to 1.
    threshold : float, optional
        The threshold value below which the maxima will not be returned. If not provided, the threshold
        is set to 50% of the maximum value in the Hough space.

    Returns
    -------
    List of tuples (x,y,r) representing the coordinates and radius of the maxima in the Hough space.
    """
    if threshold is None:
        threshold = 0.5 * np.max(hough_space)

    # find coordinates of values above the threshold
    y_idxs, x_idxs, r_idxs = np.where(hough_space >= threshold)

    # sort the coordinates based on the hough space value (descending)
    hough_vals = hough_space[y_idxs, x_idxs, r_idxs]
    sorted_idxs = np.argsort(hough_vals)[::-1]

    # get the top num_maxima coordinates
    maxima = []
    for i in range(num_maxima):
        idx = sorted_idxs[i]
        x = x_idxs[idx]
        y = y_idxs[idx]
        r = r_idxs[idx]
        maxima.append((x, y, r))

    return maxima

NUM_RADIUS_BINS = 30        # Bins are the cells or "vote counters" for each dimension in the output space
MIN_RADIUS = 20            # Circle to be detected should be greater than min radius
MAX_RADIUS = 30            # Circle to be detected should be less than max radius
NUM_MAXIMA = 35              # Number of most significant circles to be found in the input image
INPUT_IMAGE = 'pizza.png'

orig_img = cv2.imread(INPUT_IMAGE)
cv2.imshow("Original Image", orig_img)
cv2.waitKey(1)

edge_img = compute_edge_image(orig_img)
cv2.imshow("Edge Image", edge_img)
cv2.waitKey(1)

height, width = edge_img.shape
output_space = hough_transform_circles(edge_img, height, width, NUM_RADIUS_BINS, MIN_RADIUS, MAX_RADIUS)

new = np.sum(output_space, axis=2)
new = (new * 255.0 / np.max(new)).astype(np.uint8)
cv2.imshow("Hough circle space", new)

circle_parameter = find_maxima(output_space, NUM_MAXIMA)
for (v, h, r) in circle_parameter:
    cv2.circle(new, (v, h), r, (0, 0, 255), 2)
    cv2.circle(orig_img, (v, h), r, (0, 0, 255), 2)

cv2.imshow("Hough circle space with circle parameters detected", new)
cv2.imshow("Original image with circles detected", orig_img)
cv2.waitKey(0)
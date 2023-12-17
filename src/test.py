#!/usr/bin/python

# Standard imports
import cv2
import numpy as np;

# Read image
im = cv2.imread("blob.jpg")

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# # Change thresholds
params.minThreshold = 0.1
params.maxThreshold = 20000


# # Filter by Area.
params.filterByArea = True
params.minArea = 10
params.maxArea = 100000

# # Filter by Circularity
params.filterByCircularity = False
params.minCircularity = 0.1
params.maxCircularity = 1000

# # Filter by Convexity
params.filterByConvexity = False
params.minConvexity = 0.01
params.maxConvexity = 1000

# # Filter by Inertia
params.filterByInertia = False
params.minInertiaRatio = 0.01
params.maxInertiaRatio = 1000

params.filterByColor = True
params.blobColor = 255


# Create a detector with the parameters
# OLD: detector = cv2.SimpleBlobDetector(params)
detector = cv2.SimpleBlobDetector_create(params)


# Detect blobs.
keypoints = detector.detect(im)
print(keypoints[0])

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
# the size of the circle corresponds to the size of blob

im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show blobs
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)
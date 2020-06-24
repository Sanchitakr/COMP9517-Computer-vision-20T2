# Task1 Hint: (with sample code for the SIFT detector)
# Initialize SIFT detector, detect keypoints, store and show SIFT keypoints of original image in a Numpy array
# Define parameters for SIFT initializations such that we find only 10% of keypoints
import sys
import numpy as np
import math
from cv2 import cv2
import matplotlib.pyplot as plt
import imutils


class SiftDetector():
    def __init__(self, norm="L2", params=None):
        self.detector = self.get_detector(params)
        self.norm = norm

    def get_detector(self, params):
        if params is None:
            params = {}
            params["n_features"] = 0
            params["n_octave_layers"] = 3
            params["contrast_threshold"] = 0.03
            params["edge_threshold"] = 10
            params["sigma"] = 1.6

        detector = cv2.xfeatures2d.SIFT_create(
            nfeatures=params["n_features"],
            nOctaveLayers=params["n_octave_layers"],
            contrastThreshold=params["contrast_threshold"],
            edgeThreshold=params["edge_threshold"],
            sigma=params["sigma"])

        return detector


# Task2 Hint:
# Upscale the image, compute SIFT features for rescaled image
# Apply BFMatcher with defined params and ratio test to obtain good matches, and then select and draw best 5 matches
# Task3 Hint: (with sampe code for the rotation)
# Rotate the image and compute SIFT features for rotated image
# Apply BFMatcher with defined params and ratio test to obtain good matches, and then select and draw best 5 matches

# image: image to rotate
# x:     x-coordinate of point we wish to rotate around
# y:     y-coordinate of point we wish to rotate around
# angle: degrees to rotate image by
#
# Returns a rotated copy of the original image

def rotate(image, x, y, angle):
    rot_matrix = cv2.getRotationMatrix2D((x, y), angle, 1.0)
    h, w = image.shape[:2]

    return cv2.warpAffine(image, rot_matrix, (w, h))

# Get coordinates of center point.
#
# image:  Image that will be rotated
# return: (x, y) coordinates of point at center of image


def get_img_center(image):
    height, width = image.shape[:2]
    center = height // 2, width // 2

    return center

# reference doc - Introduction to SIFT: https://docs.opencv.org/3.4/da/df5/tutorial_py_sift_intro.html
# Feature matching:  https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html
def q1(image):
    sd = SiftDetector()
    # a) Extract SIFT features with default parameters and show the keypoints on the image
    keyPoints = sd.detector.detect(image, None)

    #print(len(keyPoints))
    # b) To achieve better visualization of the keypoints, reduce the number of keypoints.
    #Hint:vary nfeatures so that the number of keypoints becomes about 10 % of all default keypoints i.e 10% of 6233 = 623
    params = {}
    params["n_features"] = 623
    params["n_octave_layers"] = 3
    #params["contrast_threshold"] = 0.03
    params["contrast_threshold"] = 0.1
    params["edge_threshold"] = 10
    #Orientation Assignment
    #params["sigma"] = 1.5
    params["sigma"] = 1.6
    sift = SiftDetector(params=params)
    keyPoints1 = sift.detector.detect(image, None)
    
    img = cv2.drawKeypoints(image, keyPoints, image)
    img2 = cv2.drawKeypoints(image, keyPoints1, image)
    
    cv2.imwrite('a.jpg', img)
    cv2.imwrite('b.jpg', img2)
    return sift


def q2(image, sift):
    # a)Enlarge the given image by a scale percentage of 115.
    scale = 115
    scale = scale/100

    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    new_dim = (width, height)
    resized = cv2.resize(image, new_dim)
    

    # b) Extract the SIFT features and show the keypoints on the scaled image using the same
    # parameter setting as for Task 1 (for the reduced number of keypoints).
    ## find the keypoints and descriptors using sift detector
    keyPoints1, des1 = sift.detector.detectAndCompute(image, None)
    # Since I have  already found keypoints, call sift.compute() which computes the descriptors 
    # from the keypoints that has been already found
    keyPoints2, des2 = sift.detector.detectAndCompute(resized, None)
    img2 = cv2.drawKeypoints(image, keyPoints1, image)
    # Hint: Brute-force matching is available in OpenCV for feature matching.
    bf_matcher = cv2.BFMatcher()
    #use Matcher.match() method to get the best matches in two images
    matches = bf_matcher.match(des1, des2)
    #matches = bf_matcher.knnMatch(des1, des2, k=2)
    # c)the keypoints in both images similar which shows that they share the same common features.

    # d) Match the SIFT descriptors of the keypoints of the scaled image with those of the original image 
    # using the nearest-neighbour distance ratio method
    # We sort them in ascending order of their distances so that best matches (with low distance) come to front
    matches = sorted(matches, key=lambda x: x.distance)

    # Show the keypoints of the 5 best-matching descriptors on both the original and the scaled image.
    img_q2=cv2.drawMatches(image, keyPoints1, resized, keyPoints2,
                           matches[:6], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img_q2), plt.show()
    cv2.imwrite('d2.jpg', img_q2)
    cv2.imwrite('b2.jpg', img2)
    
    




def q3(image, sift1):
    center = get_img_center(image)
    # a) Rotate the given image clockwise by 60 degrees.
    rot = imutils.rotate_bound(image, 60)
    #rot = rotate(image, center[0], center[1], 60)
    plt.imshow(rot), plt.show()
    # b) Extract the SIFT features and show the keypoints on the rotated image using the same
    # parameter setting as for Task 1 (for the reduced number of keypoints).
    keyPoints1, des1 = sift1.detector.detectAndCompute(image, None)
    keyPoints2, des2 = sift1.detector.detectAndCompute(rot, None)

    bf = cv2.BFMatcher()
    # c)the keypoints in both images similar which shows that they share the same common features.

    # d) Match the SIFT descriptors of the keypoints of the rotated image with those of the original
    #image using the nearest-neighbour distance ratio method
    matches = bf.match(des1, des2)
    # We sort them in ascending order of their distances so that best matches (with low distance) come to front
    matches = sorted(matches, key=lambda x: x.distance)

    # Show the keypoints of the 5 best-matching descriptors on both the original and the scaled image.
    img_q3 = cv2.drawMatches(
        image, keyPoints1, rot, keyPoints2, matches[:7], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img_q3), plt.show()
    img3 = cv2.drawKeypoints(image, keyPoints1, image)
    cv2.imwrite('b3.jpg', img3)
    cv2.imwrite('d3.jpg', img_q3)
    


### the keypoints in both graph are pretty similar. it indicates that they share the same common features.
if __name__ == "__main__":

    img = cv2.imread('lab2.jpg')
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = q1(image)
    q2(image,sift)
    q3(image, sift)
    

from cv2 import cv2
import numpy as np
import argparse
import os

def maxFilter(img,  kernal_N):
    max_filter = np.zeros([img.shape[0], img.shape[1]], dtype=np.uint8)
    ksize = int((kernal_N-1)/2)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            heightLowBound = max(0, x-ksize)
            heightUpperBound = min(img.shape[0]-1, x+ksize)
            widthLowBound = max(0, y-ksize)
            widthUpperBound = min(img.shape[1]-1, y+ksize)
            max_filter[x][y] = np.max(
                img[heightLowBound:heightUpperBound, widthLowBound:widthUpperBound])
    return max_filter


def minFilter(img,  kernal_N):
    min_filter = np.zeros([img.shape[0], img.shape[1]], dtype=np.uint8)
    ksize = int((kernal_N-1)/2)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            heightLowBound = max(0, x-ksize)
            heightUpperBound = min(img.shape[0]-1, x+ksize)
            widthLowBound = max(0, y-ksize)
            widthUpperBound = min(img.shape[1]-1, y+ksize)
            min_filter[x][y] = np.min(
                img[heightLowBound:heightUpperBound, widthLowBound:widthUpperBound])
    return min_filter


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="path to input image")
    ap.add_argument("-n", '--number', required=True,
                    help="kernal size for filter")
    #free parameter, named M
    ap.add_argument('-m', '--max', required=True, help="reverse or not")

    args = vars(ap.parse_args())
    N = int(args['number'])
    if(N < 1 or N % 2 == 0):
        exit("Error: kernal filter size should be odd or positive")

    img = cv2.imread(args["image"],0)
    
#If the user sets M = 0, the algorithm should perform max-filtering(image I to A), then min-filtering(image A to B),
#then subtraction(O=I – B). And if the user sets M = 1, the algorithm should perform first
#min-filtering, then max-filtering, then subtraction.

    if(int(args["max"]) == 0):
        max_filter = maxFilter(img,  N).astype(np.uint8)
        min_filter = minFilter(max_filter, N).astype(np.uint8)
        background = cv2.subtract(img, max_filter) + 255
    else:
        min_filter = minFilter(img,  N).astype(np.uint8)
        max_filter = maxFilter(min_filter,  N).astype(np.uint8)
        background = cv2.subtract(img, max_filter)
    cv2.imshow('B', min_filter/np.max(min_filter))
    cv2.imwrite('B_task3.png', min_filter.astype(np.uint8))
    cv2.imwrite('task3.png', background.astype(np.uint8))
    cv2.imshow('task3', background/np.max(background))

    
    cv2.waitKey(0)

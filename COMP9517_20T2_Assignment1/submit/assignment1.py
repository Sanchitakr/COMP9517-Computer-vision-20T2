from cv2 import cv2
import numpy as np


def maxFilter(img,  windowsize):
    
    max_filter = np.zeros([img.shape[0], img.shape[1]], dtype=np.uint8)
    ksize = int((windowsize-1)/2)
    #windowSize = np.array(13, 13)
    #print(image.shape[0], " ", image.shape[1])
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            # getting the sliding window for the image dimension
            heightLowBound = max(0, x-ksize)
            #print("height low bound", heightLowBound)
            heightUpperBound = min(img.shape[0]-1, x+ksize)
            #print("height high bound", heightHighBound)
            widthLowBound = max(0, y-ksize)
            widthUpperBound = min(img.shape[1]-1, y+ksize)

            #window = (image[y:y + windowSize[1], x:x + windowSize[0]])
            # going through each pixel of the window to replace the MAX_VALUE in place of (x,y) of input image
            # find the maximum gray value in a neighbourhood around that pixel, and write that maximum gray value in the corresponding pixel location(x, y) in A
            MAX_VALUE = 0
            for window_x in range(heightLowBound, heightUpperBound+1):
                for window_y in range(widthLowBound, widthUpperBound+1):
                    value = img[window_x][window_y]
                    if(value > MAX_VALUE):
                        MAX_VALUE = value
            max_filter[x][y] = MAX_VALUE
    return max_filter


def minFilter(img,  windowsize):
    min_filter = np.zeros([img.shape[0], img.shape[1]], dtype=np.uint8)
    ksize = int((windowsize-1)/2)

    #Now let the algorithm go through the pixels of A one by one, and for each pixel(x, y) 
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            heightLowBound = max(0, x-ksize)
            heightUpperBound = min(img.shape[0]-1, x+ksize)
            widthLowBound = max(0, y-ksize)
            widthUpperBound = min(img.shape[1]-1, y+ksize)
            MIN_VALUE = 999
            ## find the minimum gray value in an N x N neighbourhood around that pixel, and write that minimum gray value in (x, y) in B.
            for window_x in range(heightLowBound, heightUpperBound+1):
                for window_y in range(widthLowBound, widthUpperBound+1):
                    value = img[window_x][window_y]
                    if(value < MIN_VALUE):
                        MIN_VALUE = value
            min_filter[x][y] = MIN_VALUE
    return min_filter


def correct_range(img, height, width):
    for i in range(height):
        for j in range(width):
            img[i][j] += 255
    return img


if __name__ == "__main__":
    img = cv2.imread("Particles.png",0)  
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    N = 13
    print("=====================================================TASK 1=======================================================================================================")

    max_filter = maxFilter(img,  N).astype(np.uint8)
    min_filter = minFilter(max_filter,  N).astype(np.uint8)
    # Now let the algorithm go through the pixels of max_filter one by one, and for each pixel(x, y) and put to B
    cv2.imshow('B', min_filter/np.max(min_filter))
    cv2.imwrite("B.png", min_filter.astype(np.uint8))

    print("=====================================================TASK 2=======================================================================================================")
    # removing the shading artifacts from I can be done simply by subtracting B pixel by pixel from I, resulting in the output image O

    sub = img - min_filter
    background = correct_range(sub, img.shape[0], img.shape[1])
    cv2.imshow('task2', background/np.max(background))
    cv2.imwrite("output.png", background.astype(np.uint8))

    print("===================================================== Normalisation =======================================================================================================")
    # normalising the image to get higher clarity
    out = cv2.imread("output.png")
    out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    norm = cv2.normalize(out, cv2.CV_64F, 0, 255, cv2.NORM_MINMAX)
    cv2.imshow('Output_norm', norm/np.max(norm))
    cv2.imwrite("Output_norm.png", norm.astype(np.uint8))

    cv2.waitKey(0)

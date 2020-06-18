from cv2 import cv2
import numpy as np
I = cv2.imread('Particles.png', 0)
A=I.copy()
def sliding_window(image, stepSize, windowSize):
# slide a window across the image
#stepsize is the shift to next pixel
#windowSize is the size of sliding window
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # get the current window
            print (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
            print ()
            max_pixel=(np.max(image[y:y + windowSize[1], x:x + windowSize[0]]))
            print(max_pixel)
            #window = (image[y:y + windowSize[1], x:x + windowSize[0]])
            print("x and y are :",x,y)
            print()
            #window[x,y]=max_pixel
            
            
(winW, winH) = (13, 13)
sliding_window(I, stepSize=1, windowSize=(winW, winH))

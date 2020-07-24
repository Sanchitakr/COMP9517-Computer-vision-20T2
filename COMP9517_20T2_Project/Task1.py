# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 12:13:49 2020

@author: sanch
"""

#PROGRAM - Cell tracking group Assignment COMP9571 UNSW
#
#USAGE - enter the folder pathway when prompted that contains the cell images
#        enter an integer that defines the cell type as promted
#        press keyboard key to move to next image in image sequence
#
#DESCRIPTION - Tracks cells and draws a bounding around each cell
#              provides number of cells in each image as an output to terminal 
#              keeps a running trajectory of the movement of cells

#REFERNCES - 1. http://cs-people.bu.edu/esaraee/mydocs/crowdTrack.pdf
#           2. https://stackoverflow.com/questions/30230592/loading-all-images-using-imread-from-a-given-folder
#           3. https://ri.cmu.edu/pub_files/2013/3/Thesis_SeungilHuh.pdf - from "Mitosis detection"
#           4. https://stackoverflow.com/questions/58751101/count-number-of-cells-in-the-image
#           5. https://stackoverflow.com/questions/11294859/how-to-define-the-markers-for-watershed-in-opencv
#           6.https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/
#           7. https://stackoverflow.com/questions/23398926/drawing-bounding-box-around-given-size-area-contour
#           8. https://stackoverflow.com/questions/59525640/how-to-center-the-content-object-of-a-binary-image-in-python
#           9.https://stackoverflow.com/questions/55693291/what-opencv-tracking-api-should-i-use-to-track-running-cells
#           10.https://stackoverflow.com/questions/56325204/difference-in-numpy-array-masking
import numpy as np
import cv2
import imutils
import os
#set a default filepath
#file_path = 'Images\DIC-C2DH-HeLa\Sequence 1'
#dataset = 1

#get inputs through user
print("Enter image folder path, e.g \"Images/DIC-C2DH-HeLa/Sequence 1\"") 
file_path = input()
print("Define cell size (DIC-C2DH-HeLa = 1, Fluo-N2DL-HeLa = 2, PhC-C2DL-PSC = 3")
dataset = int(input())
#------------------------------------------ LOGIC ----------------------------------------------------------

#open first image in file to get size for the cell tracking pathway map
#assumes first file in folder is called "t000.tif"
(ht,wd,dt) = cv2.imread(file_path+"/t000.tif").shape
pathway_mask = np.zeros((ht,wd,3), np.uint8)
pathway_color = [255, 0, 0]

#loop to open all the images in the given filepath and perform tracking on them

directory = os.fsencode(file_path)
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    
    img = cv2.imread(str(file_path+"/"+filename))

    #define elipse based kernel as base
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,  (15,15))
    
    #different preproccessing techniques for different cells
    if(dataset == 3):
        tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
        gray = cv2.cvtColor(tophat, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #gaussian blur and then otsu thresholding to reduce grey noise in the cells and improve watershed segmentation
    #results
    blur = cv2.GaussianBlur(gray,(3,3),0)
    ret,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    print ("threshold s :",thresh)

    #get rid of artifacts that are too small to be cells
    #from labs
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,  (5,5))
    thresh = cv2.erode(thresh,kernel,iterations=1)
    thresh = cv2.dilate(thresh,kernel,iterations=1)
    thresh = cv2.dilate(thresh,kernel,iterations=1)
    thresh = cv2.erode(thresh,kernel,iterations=1)

    #perform erode on the PhC-C2DL-PSC dataset for better sepperation of close cells - more clarity
    if dataset == 3:
        thresh = cv2.erode(thresh,kernel,iterations=1)
        

    #perform watershedding segmentation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,  (3,3))
    background = cv2.dilate(thresh,kernel,iterations=1)
    uncertain_area = cv2.subtract(background,thresh)
    #computes the connected components labeled image of boolean image
    ret, markers = cv2.connectedComponents(thresh)

    markers = markers+1
    markers[uncertain_area==255] = 0
    
    markers = cv2.watershed(img,markers)
    img[markers == -1] = [255,0,0]

    unique_markers = np.unique(markers)
    
    #print the number of cells in the image to terminal output
    #need to subtract 2 to get number of cells since background is counted twice
    print("number of cells in image("+filename+"): "+ str(len(unique_markers)-2))

    #iterate over segmented markers and draw bounding boxes for each cell
    #uses the centroid of the bound to show cell trajectory on a sepperate mask
    #converting the image to HSV format then use a lower/upper color threshold to create a binary mask. 
    #From here we perform morphological operations to smooth the image and remove small bits of noise.
    
    for m in unique_markers:
        if m == -1 or m == 1:
            continue
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[markers == m] = 255
        #Now that we have the mask, we find contours with the cv2.RETR_EXTERNAL parameter
        contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        c = max(contours, key=cv2.contourArea)
        #create bounding box for cell - tracking
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #show centroid of the cells bounding box on the pathway image - tracking
        pathway_mask[int(y+h/2),int(x+w/2)] = (pathway_color[0],pathway_color[1],pathway_color[2])
        
    #update the pathway color to differentiate between timesteps on the pathway mask
    if (pathway_color[1] != 255):
        pathway_color[1] = pathway_color[1]+1
    elif (pathway_color[2] != 255):
        pathway_color[2] = pathway_color[2]+1
    else:
        pathway_color = (255, 0 ,0)

    #show the results on current image
    #press any key to move to next image
    #shows an ongoing cell trajectory
    cv2.imshow("thresh", thresh)
    cv2.imshow("img", img)
    cv2.imshow("centroids", pathway_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
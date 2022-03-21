import sys
import math
import cv2
import numpy as np
import time
from matplotlib import pyplot as plt
from scipy.ndimage.filters import maximum_filter1d





# padding the image
def padding(image):
    height, width = image.shape
    pad = 3

    P1 = height+2*pad
    P2 = width+2*pad

    padded_image = np.zeros((P1, P2))
    padded_image[pad:-pad,pad:-pad] = image

    return padded_image





# implement adaptive median filtering
def adaptiveMedianFilter(image):
    # get the image height and width
    h, w = image.shape

    s = 3
    sMax = 21
    a = sMax//2

    # pad the image
    padded_image = padding(image)

    # initialize new image in correct size
    f_image = np.zeros(padded_image.shape)
    
    # loop through the image at the correct size
    for i in range(a,h-1):
        for j in range(a,w-1):
            # call stage A for every pixel in the image 
            f_image[i,j] = stageA(padded_image, i,j,s, sMax)
    
    return f_image[a:-a,a:-a] 





# code from slides
# takes the image, the pixel location, the filter size, and the max filter size
def stageA(mat,x,y,s,sMax):

    window = mat[x-(s//2):x+(s//2)+1,y-(s//2):y+(s//2)+1]

    # gets minimum, maximum, and median gray level
    try:
        Zmin = np.min(window)
    except ValueError:  
        Zmin = 0
    Zmed = np.median(window)
    Zmax = np.max(window)

    # calculate A1 and A2
    A1 = Zmed - Zmin
    A2 = Zmed - Zmax

    if(A1 > 0 and A2 < 0):
        return stageB(window)
    else:
        s +=2
        if (s <= sMax):
            return stageA(mat,x,y,s,sMax)
        else:
            return Zmed





# code from slides
def stageB(window):
    h,w = window.shape
    Zmin = np.min(window)
    Zmed = np.median(window) 
    Zmax = np.max(window)

    Zxy = window[h//2,w//2]
    B1 = Zxy - Zmin
    B2 = Zxy - Zmax

    if (B1 > 0 and B2 < 0):
        return Zxy
    else:
        return Zmed





# main function
def main():

    # read grayscale
    image = cv2.imread(sys.argv[1], 0)

    # display original image
    cv2.imshow('Original Image', image)
    cv2.waitKey(0)

    # letting the user know the program is running
    print("Implementing Adaptive Median Filter...")

    # adaptive median filter with timing
    start_time = time.perf_counter()
    newImage = adaptiveMedianFilter(image)
    end_time = time.perf_counter()

    # displaying image in the right format
    newImage = newImage.astype(np.uint8)
    cv2.imshow('Adaptive Median Filter Image !', newImage)
    cv2.waitKey(0) 

    # display the time
    print(f"\n\nThe Adaptive Median Filter took {end_time - start_time:0.4f} seconds for this image!\n")


main()

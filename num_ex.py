import cv2
import numpy as np
import matplotlib.pyplot as plt

def show(img):
    plt.imshow(img, cmap="gray")
    plt.show()

def ocr(img):
    # Tesseract mode settings:
    #   Page Segmentation mode (PSmode) = 3 (defualt = 3)
    #   OCR Enginer Mode (OEM) = 3 (defualt = 3)
    tesser = cv2.text.OCRTesseract_create('/usr/local/Cellar/tesseract/4.1.0/share/tessdata/','eng','0123456789',3,3)
    retval = tesser.run(img, 0) # return string type
    print ('OCR Output: ' + retval)

# Directly feed image to Tesseact
img = cv2.imread('num_test.png')
ocr(img)

# Load image as gray scale 
img = cv2.imread('num_test.png',0)
show(img)
ocr(img)

# Enhance image and get same positive result
ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY) 
kernel = np.ones((3,3),np.uint8)
img = cv2.erode(thresh,kernel,iterations = 1)
show(img)
ocr(img)

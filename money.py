import cv2
import matplotlib.pyplot as plt
import numpy as np
import operator
import os

# module level variables ##########################################################################
MIN_CONTOUR_AREA = 125
RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

###################################################################################################
class ContourWithData():

    # member variables ############################################################################
    npaContour = None           # contour
    boundingRect = None         # bounding rect for contour
    intRectX = 0                # bounding rect top left corner x location
    intRectY = 0                # bounding rect top left corner y location
    intRectWidth = 0            # bounding rect width
    intRectHeight = 0           # bounding rect height
    fltArea = 0.0               # area of contour

    def calculateRectTopLeftPointAndWidthAndHeight(self):               # calculate bounding rect info
        [intX, intY, intWidth, intHeight] = self.boundingRect
        self.intRectX = intX
        self.intRectY = intY
        self.intRectWidth = intWidth
        self.intRectHeight = intHeight

    def checkIfContourIsValid(self):                            # this is oversimplified, for a production grade program
        if self.fltArea < MIN_CONTOUR_AREA: return False        # much better validity checking would be necessary
        return True

def get_text(imgTestingNumbers):
    allContoursWithData = []                # declare empty lists,
    validContoursWithData = []              # we will fill these shortly

    
    ######################################################################
    imgThresh = imgTestingNumbers
    imgThreshCopy = imgThresh.copy()        # make a copy of the thresh image, this in necessary b/c findContours modifies the image
    

    #imgContours, 
    npaContours, npaHierarchy = cv2.findContours(imgThreshCopy,             # input image, make sure to use a copy since the function will modify this image in the course of finding contours
                                                 cv2.RETR_EXTERNAL,         # retrieve the outermost contours only
                                                 cv2.CHAIN_APPROX_SIMPLE)   # compress horizontal, vertical, and diagonal segments and leave only their end points

    for npaContour in npaContours:                             # for each contour
        contourWithData = ContourWithData()                                             # instantiate a contour with data object
        contourWithData.npaContour = npaContour                                         # assign contour to contour with data
        contourWithData.boundingRect = cv2.boundingRect(contourWithData.npaContour)     # get the bounding rect
        contourWithData.calculateRectTopLeftPointAndWidthAndHeight()                    # get bounding rect info
        contourWithData.fltArea = cv2.contourArea(contourWithData.npaContour)           # calculate the contour area
        allContoursWithData.append(contourWithData)                                     # add contour with data object to list of all contours with data
    # end for

    for contourWithData in allContoursWithData:                 # for all contours
        if contourWithData.checkIfContourIsValid():             # check if valid
            validContoursWithData.append(contourWithData)       # if so, append to valid contour list
        # end if
    # end for

    validContoursWithData.sort(key = operator.attrgetter("intRectX"))         # sort contours from left to right

    strFinalString = ""         # declare final string, this will have the final number sequence by the end of the program

    for contourWithData in validContoursWithData:            # for each contour
        imgROI = imgThresh[contourWithData.intRectY : contourWithData.intRectY + contourWithData.intRectHeight,     # crop char out of threshold image
                           contourWithData.intRectX : contourWithData.intRectX + contourWithData.intRectWidth]

        imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))             # resize image, this will be more consistent for recognition and storage

        npaROIResized = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))      # flatten image into 1d numpy array

        npaROIResized = np.float32(npaROIResized)       # convert from 1d numpy array of ints to 1d numpy array of floats

        retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k = 5)     # call KNN function find_nearest

        strCurrentChar = str(chr(int(npaResults[0][0])))                                             # get character from results

        strFinalString = strFinalString + strCurrentChar            # append current char to full string
    # end for

    #print (strFinalString)                  # show the full string
    return int(strFinalString)

###################################################################################################

cap = cv2.VideoCapture('./bar_test.mp4')
shot = np.zeros((50, 160))
frame_cntr = 0#frame counter to exract 
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))#erosion kernal
############################################################
#initialize the knn model
npaClassifications = np.loadtxt("classifications.txt", np.float32)                  # read in training classifications
npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)                 # read in training images
npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))       # reshape numpy array to 1d, necessary to pass to call to train
kNearest = cv2.ml.KNearest_create()                   # instantiate KNN object
kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)
############################################################
money_stack = []
mean_money = 0
while(cap.isOpened()):
    ret, img = cap.read()
    if (ret == False):
        break
    else:
        frame_cntr = frame_cntr+1
        money_region = img[1045:1070, 1200:1280]
        money_region = cv2.resize(money_region, (160, 50))
        money_region_gray = cv2.cvtColor(money_region, cv2.COLOR_BGR2GRAY)
        _, money_region_binary = cv2.threshold(money_region_gray, 
                                                100, #threshold = 200
                                                255, #max value in image
                                                cv2.THRESH_BINARY)#binary thresh
        #######got the money########
        money = money_region_binary
        erosion = cv2.erode(money,kernel)
        closing = cv2.morphologyEx(erosion, cv2.MORPH_CLOSE, kernel)
        cv2.imshow('money',erosion)
        if (frame_cntr <= 30):
            money_stack.append(get_text(closing))
        if (frame_cntr == 30):
            frame_cntr = 0 
            max_id = money_stack.index(max(money_stack))
            min_id = money_stack.index(min(money_stack))
            mean_money = 0
            for i in range(30):
                if (i != max_id)and(i != min_id):
                    mean_money = mean_money + money_stack[i]
            mean_money = int(mean_money/28)
            print(mean_money)
            money_stack = []
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract as pt

def bars_region(image): 
    #get the resolution of the image
    height, width, channel = image.shape
    area = [(width*0.25, height),(width*0.25, height*0.9),(width*0.7, height*0.9),(width*0.7, height),]
    crop_area = np.array([area], np.int32)
    #set the background of the mask to 0
    mask = np.zeros_like(image)
    #get the mask done, the mask only allows minimap area to be further processed
    cv2.fillPoly(mask, crop_area, (255,255,255))
    bottom_region = cv2.bitwise_and(image, mask)
    return bottom_region

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 1)
    return line_image

def get_bar(lines):
    a = 1920
    b = 1080
    c = 0
    d = 0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            #get verdical/horizontal lines of the health bar
            if (abs(y1-y2)<=2):#horizontal boudary
                a = min(x1,x2) if (a > min(x1,x2)) else a
                c = max(x1,x2) if (c < max(x1,x2)) else c
                b = y1 if (b > y1) else b
                d = y1 if (d < y1) else d
    return [a,b,c,d]

def extracting_health(image_RGB, bottom_region):
    #setting green health bar binning parameters
    low_green = np.array([0,60,0])
    high_green = np.array([30,255,70])
    #color select the green health bar
    health_bar = cv2.inRange(bottom_region, low_green, high_green)
    #hough transformation
    rho = 2
    theta = np.pi/180
    threshold = 100
    lines = cv2.HoughLinesP(health_bar,rho, theta, threshold, np.array ([]), minLineLength=60, maxLineGap=10)
    #checking
    line_image = display_lines(health_bar, lines)
    line_image = cv2.cvtColor(line_image,cv2.COLOR_GRAY2RGB)
    health_bar_show_in_image = cv2.addWeighted(image_RGB,0.8,line_image,1,1)
    result = cv2.resize(health_bar_show_in_image, (1920,1080))
    '''
    plt.figure()
    plt.imshow(result)
    '''
    #checking
    if lines is not None:
        return get_bar(lines)
    else:
        return None

def extracting_mana(image_RGB, bottom_region):
    #setting green health bar binning parameters
    low_blue = np.array([0,50,84])
    high_blue = np.array([60,150,255])
    #color select the green health bar
    mana_bar = cv2.inRange(bottom_region, low_blue, high_blue)
    #hough transformation
    rho = 2
    theta = np.pi/180
    threshold = 100
    lines = cv2.HoughLinesP(mana_bar,rho, theta, threshold, np.array ([]), minLineLength=60, maxLineGap=10)
    #checking
    line_image = display_lines(mana_bar, lines)
    line_image = cv2.cvtColor(line_image,cv2.COLOR_GRAY2RGB)
    mana_bar_show_in_image = cv2.addWeighted(image_RGB,0.8,line_image,1,1)
    result = cv2.resize(mana_bar_show_in_image, (1920,1080))
    '''
    plt.figure()
    plt.imshow(result)
    '''
    #checking
    if lines is not None:
        return get_bar(lines)
    else:
        return None

def get_text(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_red = img[:,:,0]
    _,img_digits = cv2.threshold(img_gray, 230, 255, cv2.THRESH_BINARY)
    img_digits = img_digits[:,150:275]
    img_digits = cv2.resize(img_digits,(500,60))
    ##################
    #insert image to text
    ##################
    #text = '340/789'
    return img_digits

class bar:
    def __init__(self, loc):
        self.loc = loc
    def get_location(self):
        return self.loc
    def write_location(self, loc):
        self.loc = loc

frame_counter = 0
cap = cv2.VideoCapture("./bar_test.mp4")
initiate = False
bars = []
while(initiate == False):
    _, img = cap.read()
    image = cv2.resize(img,(1920, 1080))
    #converting image to RGB space from BGR
    image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #spotting on the bottom region
    bottom_region = bars_region(image_RGB)
    h_co = extracting_health(image_RGB,bottom_region) 
    if (h_co != None):
        bars.append(h_co)

    m_co = extracting_mana(image_RGB,bottom_region) 
    if (m_co != None):
        bars.append(m_co)
    #get the numbers
    health = get_text(image_RGB[h_co[1]:h_co[3],h_co[0]:h_co[2]])
    initiate = True
    frame_counter = frame_counter + 1
    #show the image
    #cv2.imshow('mana digits',mana)
    #cv2.waitKey(0)
longshot = np.zeros_like(health)
#check the bar state
print(bars)
print('__________')
while(cap.isOpened()):
    ret, frame = cap.read()
    frame_counter = frame_counter + 1
    if (ret == False):
        break
    else:
        for bar in bars:
            print(bar)
            bar_img = frame[bar[1]:bar[3],bar[0]:bar[2]]
            cv2.imshow('p',bar_img)
            if (frame_counter%15 == 0)and(frame_counter > 100):
                longshot = np.concatenate((longshot, get_text(bar_img)), axis=0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.imwrite('./health_bar_shot/health.png',longshot)
cap.release()
cv2.destroyAllWindows()